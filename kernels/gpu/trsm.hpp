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

#ifndef REMORA_KERNELS_CLBLAS_TRSM_HPP
#define REMORA_KERNELS_CLBLAS_TRSM_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <boost/compute/functional/operator.hpp> //for multiplies
///solves systems of triangular matrices

namespace remora {namespace bindings {
struct trsm_kernel{
	boost::compute::kernel kernel;
	std::size_t K_index;
	std::size_t start_index;
	std::size_t end_index;
	std::size_t unit_index;
	std::size_t upper_index;
};
//Lower triangular - matrix(row-major)
template<class MatA, class MatB>
trsm_kernel createTRSMDiagBlockKernel(
	matrix_expression<MatA, gpu_tag> const& A_unreg,
	matrix_expression<MatB, gpu_tag>& B_unreg,
	char const* options
){
	typedef typename MatA::value_type value_typeA;
	typedef typename MatB::value_type value_typeB;
	boost::compute::multiplies<value_typeB> prod;
	
	gpu::detail::meta_kernel k("blas_trsm");
	std::size_t K_index = k.add_arg<std::size_t>("K");//number of columns in B
	std::size_t start_index = k.add_arg<std::size_t>("start");//start of block of A
	std::size_t end_index = k.add_arg<std::size_t>("end");//end of Block of A
	std::size_t unit_index = k.add_arg<std::size_t>("unit");//whether A is unit triangular
	std::size_t upper_index = k.add_arg<std::size_t>("upper");//whether A is upper triangular
	auto A = k.register_args(to_functor(A_unreg));
	auto B = k.register_args(to_functor(B_unreg));
	// Local memory to fit a tile of A and B
	// we store B as column major in local memory
	// we also allocate memory to store results of B
	k << "__local " <<k.decl<value_typeA>("Asub")<< "[TILE_SIZE][TILE_SIZE+2];\n";//+2 to avoid bank conflicts
	k << "__local " <<k.decl<value_typeB>("Bsub")<< "[TILE_SIZE_K][TILE_SIZE+2];\n";//+2 to avoid bank conflicts
	k << "const ulong numWorkers = get_local_size(0);\n";
	//ensure we are not reading out of bounds
	k << "const ulong t = get_group_id(1);\n";
	k << "const ulong curTileA = end-start;\n";
	k << "const ulong curTileK =  min(TILE_SIZE_K, K - t*TILE_SIZE_K);\n";
	
	// Load tile of A into local memory
	k << "for(ulong i = get_local_id(0); i < TILE_SIZE; i += numWorkers){\n";
	k << "	for(ulong j = get_local_id(1); j < TILE_SIZE; j += numWorkers){\n";
	k << "		Asub[i][j] ="<< A(k.expr<cl_ulong>("min(end-1, start + i)"),k.expr<cl_ulong>("min(end-1, start + j)"))<<";\n";
	k << "	}\n";
	k << "}\n";
	
	
	// Load Tile of B into local memory, store columns of B as rows
	k << "for(ulong i = get_local_id(0); i < TILE_SIZE; i += numWorkers){\n";
	k << "	for(ulong k = get_local_id(1); k < TILE_SIZE_K; k += numWorkers){\n";
	k << "		Bsub[k][i] ="<< B(k.expr<cl_ulong>("min(end-1,start + i)"),k.expr<cl_ulong>("min(K-1,t * TILE_SIZE_K+k)"))<<";\n";
	k << "	}\n";
	k << "}\n";
	// Synchronise to make sure the tiles are loaded
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";

	// Loop over the values of a single tile
	//lower-case
	k << "if(!upper){\n";
	k << "	for(ulong k = get_local_id(1); k < curTileK; k += numWorkers){\n";
	k << "		for(ulong i = 0; i < TILE_SIZE && get_local_id(0) == 0; ++i){\n";
	k << "			if(!unit){Bsub[k][i] /= Asub[i][i];}\n";
	k << "			for(ulong j = i+1; j < TILE_SIZE; ++j){\n";
	k << "				Bsub[k][j] -= "<< prod(k.expr<value_typeB>("Bsub[k][i]"), k.expr<value_typeA>("Asub[j][i]"))<<";\n";
	k << "			}\n";
	k << "		}\n";
	k << "	}\n";
	k << "}else{\n";
	//upper case
	k << "	for(ulong k = get_local_id(1); k < curTileK; k += numWorkers){\n";
	k << "		for(ulong n = curTileA; n > 0 && get_local_id(0) == 0; --n){\n";
	k << "			ulong i = n-1;\n";
	k << "			if(!unit ){Bsub[k][i] /= Asub[i][i];}\n";
	k << "			for(ulong j = 0; j < i; j ++){\n";
	k << "					Bsub[k][j] -= "<< prod(k.expr<value_typeB>("Bsub[k][i]"), k.expr<value_typeA>("Asub[j][i]"))<<";\n";
	k << "			}\n";
	k << "		}\n";
	k << "	}\n";
	k << "}\n";
	// Synchronise before continuing
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";
	// Store the final results back in B
	k << "for(ulong i = get_local_id(0); i < curTileA; i += numWorkers){\n";
	k << "	for(ulong k = get_local_id(1); k < curTileK; k += numWorkers){\n";
	k << B(k.expr<cl_ulong>("(start+i)"),k.expr<cl_ulong>("(t * TILE_SIZE_K+k)"))<<" =  Bsub[k][i];\n";
	k << "	}\n";
	k << "}\n";
	
	boost::compute::kernel kernel = k.compile(B_unreg().queue().get_context(), options);
	return {kernel,K_index,start_index,end_index,unit_index,upper_index};
}

template <typename MatA, typename MatB, class Triangular>
void trsm_recursive(
	matrix_expression<MatA, gpu_tag> const& Afull, 
	matrix_expression<MatB, gpu_tag> & Bfull,
	trsm_kernel& kernel,
	std::size_t start,
	std::size_t end,
	std::size_t tileSizeA,
	std::size_t tileSizeB,
	std::size_t numWorkers,
	Triangular t
){
	auto A = subrange(Afull,start,end,start,end);
	auto B = rows(Bfull,start,end);
	std::size_t size = A.size1();
	//if the matrix is small enough call the computation kernel directly for the block
	if(size <= tileSizeA){
		//enqueue kernel with kernel args
		kernel.kernel.set_arg(kernel.K_index, Bfull().size2());
		kernel.kernel.set_arg(kernel.start_index, start);
		kernel.kernel.set_arg(kernel.end_index, end);
		kernel.kernel.set_arg(kernel.unit_index, (std::size_t)Triangular::is_unit);
		kernel.kernel.set_arg(kernel.upper_index, (std::size_t)Triangular::is_upper);
		
		std::size_t global_work_size[2] = {
			numWorkers,
			(Bfull().size2()+tileSizeB-1)/ tileSizeB * numWorkers
		};
		std::size_t local_work_size[2] = {numWorkers, numWorkers};
		Bfull().queue().enqueue_nd_range_kernel(kernel.kernel, 2,nullptr, global_work_size, local_work_size);
		return;
	}
	std::size_t numBlocks = (A.size1()+tileSizeA-1)/tileSizeA;
	std::size_t split = numBlocks/2 * tileSizeA;
	auto Bfront = rows(B,0,split);
	auto Bback = rows(B,split,size);
	
	//otherwise run the kernel recursively
	if(Triangular::is_upper){ //Upper triangular case
		trsm_recursive(Afull, Bfull, kernel, start+split,end, tileSizeA,tileSizeB, numWorkers, t);
		kernels::gemm(subrange(A,0,split,split,size), Bback, Bfront, -1.0);
		trsm_recursive(Afull, Bfull, kernel, start,start+split, tileSizeA,tileSizeB, numWorkers, t);
	}else{// Lower triangular caste
		trsm_recursive(Afull, Bfull, kernel, start,start+split, tileSizeA,tileSizeB, numWorkers, t);
		kernels::gemm(subrange(A,split,size,0,split), Bfront, Bback, -1.0);
		trsm_recursive(Afull, Bfull, kernel, start+split,end, tileSizeA,tileSizeB, numWorkers, t);
	}
}

template <typename MatA, typename MatB, class Triangular>
void trsm_call(
	matrix_expression<MatA, gpu_tag> const& A, 
	matrix_expression<MatB, gpu_tag>& B,
	Triangular,
	left
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size2() == B().size1());
	std::size_t const TileSizeA = 32;//size of the diagonal blocks where the single kernel runs
	std::size_t const TileSizeB = 32;// size of the blocks B is partitioned into along the number of columns
	std::size_t const numWorkers = 8; //number of workers in two dimensions (e.g. 8x8=64)
	char const* options ="-DTILE_SIZE=32ul -DTILE_SIZE_K=32ul";
	auto kernel = bindings::createTRSMDiagBlockKernel(A,B,options);
	
	trsm_recursive(A,B,kernel,0,A().size1(), TileSizeA, TileSizeB, numWorkers,Triangular());
}

template <typename MatA, typename MatB, class Triangular>
void trsm_call(
	matrix_expression<MatA, gpu_tag> const& A, 
	matrix_expression<MatB, gpu_tag>& B,
	Triangular,
	right
){
	auto transB = trans(B);
	trsm_call(trans(A),transB,typename Triangular::transposed_orientation(),left());
}

}
namespace kernels{
//main kernel runs the kernel above recursively and calls gemv
template <class Triangular, class Side, typename MatA, typename MatB>
void trsm(
	matrix_expression<MatA, gpu_tag> const& A, 
	matrix_expression<MatB, gpu_tag>& B
){
	bindings::trsm_call(A,B,Triangular(), Side());
}
}}
#endif
