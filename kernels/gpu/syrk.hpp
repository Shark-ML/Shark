//===========================================================================
/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2016
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#ifndef REMORA_KERNELS_CLBLAS_SYRK_HPP
#define REMORA_KERNELS_CLBLAS_SYRK_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <boost/compute/functional/operator.hpp> //for multiplies

namespace remora{ namespace kernels{

// C <- alpha * A * B + beta * C
template <bool Upper, typename MatA, typename MatC>
void syrk(
	matrix_expression<MatA, gpu_tag> const& A_unreg,
	matrix_expression<MatC, gpu_tag>& C_unreg, 
	typename MatC::value_type const& alpha
) {
	REMORA_SIZE_CHECK(A_unreg().size1() == C_unreg().size1());
	REMORA_SIZE_CHECK(C_unreg().size1()== C_unreg().size2());
	
	// TUNING VARIABLES:
	// TILE_SIZE: width and height of a tile computed in C
	// TILE_SIZE_K: length of Tile loaded from A, i.e A is loaded as (TILE_SIZE, TILE_SIZE_K) matrix
	// BLOCK_SIZE: size of a subblock computed by a single work item (must divide TILE_SIZE)
	// it holds TILE_SIZE = get_local_size(i) * BLOCK_SIZE for i=0,1
	// note that BLOCK_SIZE increases the number of registers needed per thread.
	// roughly O(BLOCK_SIZE^2) registers are needed per thread (+ overhead). Note that register spill might be deadly to performance.
	// local memory is TILE_SIZE*TILE_SIZE_K*2*sizeof(MatC::value_type)
	// increasing TILE_SIZE improves the trade-off "computations per memory access"
	std::size_t BLOCK_SIZE = 4;
	std::size_t TILE_SIZE = 32;
	std::size_t NUM_WORKERS = TILE_SIZE / BLOCK_SIZE;
	char const* options ="-DTILE_SIZE=32ul -DBLOCK_SIZE=4ul -DTILE_SIZE_K=16ul";
	typedef typename MatC::value_type value_type;
	
	gpu::detail::meta_kernel k("blas_syrk");
	std::size_t N_index = k.add_arg<std::size_t>("N");
	std::size_t K_index = k.add_arg<std::size_t>("K");
	std::size_t upper_index = k.add_arg<std::size_t>("upper");
	std::size_t alpha_index = k.add_arg<value_type>("alpha");
	auto A = k.register_args(to_functor(A_unreg));
	auto C = k.register_args(to_functor(C_unreg));
	//check whether we are in a block that is not touched by syrk
	k <<"if((upper && get_group_id(1) < get_group_id(0))) return;\n"; 
	k <<"if((!upper && get_group_id(1) > get_group_id(0))) return;\n"; 
	
	// From now on this the normal gemm kernel with the difference that A and B are the same (but transposed) matrices.
	// Local memory to fit a tile of A and B
	// we transpose A locally in memory
	k << "__local " <<k.decl<value_type>("Asub")<< "[TILE_SIZE_K][TILE_SIZE+2];\n";//+2 to avoid bank conflicts
	k << "__local " <<k.decl<value_type>("Bsub")<< "[TILE_SIZE_K][TILE_SIZE+2];\n";//+2 to avoid bank conflicts
	k << "	const ulong numWorkers = get_local_size(0);\n";
	// Initialise the accumulation registers
	// here the subblock of C for this thread is stored
	// blocks ae not continuous but strided so that
	// coalesced write to C is possible
	// e.g. with 8x8 threads and BLOCK_SIZE 2x2 thread 1 has local tile elements
	//(0,0) (0,8) (8,0), (8,8). all other blocks are calculated
	// by adding (local_id(0), local_id(1)) to the coordinates
	k << k.decl<value_type>("acc") <<"[BLOCK_SIZE][BLOCK_SIZE];\n";
	k << "for (ulong wm=0; wm<BLOCK_SIZE; wm++){\n";
	k << "	for (ulong wn=0; wn<BLOCK_SIZE; wn++){\n";
	k << "		acc[wm][wn] = 0.0f;\n";
	k << "	}\n";
	k << "}\n";
	

	// Loop over all tiles
	k << "ulong numTiles = (K+TILE_SIZE_K-1)/TILE_SIZE_K;\n";
	k << "for (ulong t=0; t<numTiles; t++){\n";
		
	//ensure we are not reading out of bounds in K.
	k << "	const ulong curTileK =  min(TILE_SIZE_K, K - t*TILE_SIZE_K);\n";
		
	// Load one tile of A and B transposed ulongo local memory using padding
	k << "	for(ulong i = get_local_id(0); i < TILE_SIZE; i += numWorkers){\n";
	k << "		for(ulong k = get_local_id(1); k < curTileK; k += numWorkers){\n";
	k << "			ulong ktile = t * TILE_SIZE_K + k;\n";
	k << "			Asub[k][i] ="<< A(k.expr<cl_ulong>("min(N-1,TILE_SIZE * get_group_id(0)+i)"), k.expr<cl_ulong>("ktile"))<<";\n";
	k << "			Bsub[k][i] ="<< A(k.expr<cl_ulong>("min(N-1,TILE_SIZE * get_group_id(1)+i)"), k.expr<cl_ulong>("ktile"))<<";\n";
	k << "		}\n";
	k << "	}\n";

	// Synchronise to make sure the tile is loaded
	k << "	barrier(CLK_LOCAL_MEM_FENCE);\n";

	// Loop over the values of a single tile
	// by computing outer products ulongo the local accumulation registers acc
	k << "	for (ulong k=0; k<curTileK; k++){\n";
	// Cache the values of Bsub in registers to save local memory lookups
	k <<  k.decl<value_type>("Breg")<<"[BLOCK_SIZE];\n";
	k << "		for (ulong wn=0; wn<BLOCK_SIZE; wn++){\n";
	k << "			Breg[wn] = Bsub[k][get_local_id(1) + wn * numWorkers];\n";
	k << "		}\n";

	// Perform the computation
	k << "		for (ulong wm = 0; wm<BLOCK_SIZE; wm++){\n";
	k << k.decl<value_type>("Areg") << "= Asub[k][get_local_id(0) + wm * numWorkers];\n";
	k << "			for (ulong wn=0; wn<BLOCK_SIZE; wn++){\n";
	k << "				acc[wm][wn] += Areg * Breg[wn];\n";
	k << "			}\n";
	k << "		}\n";
	k << "	}\n";

	// Synchronise before loading the next tile
	k << "	barrier(CLK_LOCAL_MEM_FENCE);\n";
	k << "}\n";

	// Store the final results in C
	k << "const ulong maxCi = min(TILE_SIZE, N -  get_group_id(0) * TILE_SIZE);\n";
	k << "const ulong maxCj = min(TILE_SIZE, N -  get_group_id(1) * TILE_SIZE);\n";
	k << "const ulong offTileCi = TILE_SIZE * get_group_id(0);\n";
	k << "const ulong offTileCj = TILE_SIZE * get_group_id(1);\n";
	k << "ulong wm = 0;\n";
	k << "for (ulong i = get_local_id(0); i < maxCi; i += numWorkers, wm++){\n";
	k << "	ulong wn = 0;\n";
	k << "	for (ulong j =get_local_id(1); j < maxCj; j += numWorkers, wn++){\n";
	k << "		if(get_group_id(1) != get_group_id(0) || (upper && j >= i) || (!upper && j <= i) ){\n";
	k <<			C(k.expr<cl_ulong>("(offTileCi + i)"), k.expr<cl_ulong>("(offTileCj + j)")) << "+= alpha * acc[wm][wn];\n";
	k << "		}\n";
	k << "	}\n";
	k << "}\n";
	
	boost::compute::kernel kernel = k.compile(C_unreg().queue().get_context(), options);
	
	//enqueue kernel with kernel args
	kernel.set_arg(N_index, C_unreg().size1());
	kernel.set_arg(K_index, A_unreg().size2());
	kernel.set_arg(alpha_index, alpha);
	kernel.set_arg(upper_index, (std::size_t)Upper);
	
	std::size_t global_work_size[2] = {
		(C_unreg().size1()+TILE_SIZE-1)/ TILE_SIZE * NUM_WORKERS,
		(C_unreg().size2()+TILE_SIZE-1)/ TILE_SIZE * NUM_WORKERS
	};
	std::size_t local_work_size[2] = {NUM_WORKERS, NUM_WORKERS};
	C_unreg().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, local_work_size);
}

}}

#endif
