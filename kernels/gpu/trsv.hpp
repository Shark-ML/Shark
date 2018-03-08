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

#ifndef REMORA_KERNELS_CLBLAS_TRSV_HPP
#define REMORA_KERNELS_CLBLAS_TRSV_HPP


#include "../../detail/traits.hpp"
#include "../../proxy_expressions.hpp"
#include "../gemv.hpp"
#include <boost/compute/functional/operator.hpp> //for multiplies

///solves systems of triangular matrices with one left hand side

namespace remora{namespace bindings {
struct trsv_kernel{
	boost::compute::kernel kernel;
	std::size_t start_index;
	std::size_t end_index;
	std::size_t unit_index;
	std::size_t upper_index;
};
//Lower triangular - matrix(row-major)
template<class MatA, class VecB>
trsv_kernel createTRSVDiagBlockKernel(
	matrix_expression<MatA, gpu_tag> const& A_unreg,
	vector_expression<VecB, gpu_tag> &b_unreg,
	char const* options
){
	typedef typename MatA::value_type value_typeA;
	typedef typename VecB::value_type value_typeB;
	boost::compute::multiplies<value_typeB> prod;
	
	gpu::detail::meta_kernel k("blas_trsv");
	std::size_t start_index = k.add_arg<std::size_t>("start");//start of block of A
	std::size_t end_index = k.add_arg<std::size_t>("end");//end of Block of A
	std::size_t unit_index = k.add_arg<std::size_t>("unit");//whether A is unit triangular
	std::size_t upper_index = k.add_arg<std::size_t>("upper");//whether A is upper triangular
	auto A = k.register_args(to_functor(A_unreg));
	auto b = k.register_args(to_functor(b_unreg));
	// Local memory to fit a tile of A and the vector B
	k << "__local " <<k.decl<value_typeA>("Asub")<< "[TILE_SIZE][TILE_SIZE+2];\n";//+2 to avoid bank conflicts
	k << "__local " <<k.decl<value_typeB>("Bsub")<< "[TILE_SIZE];\n";
	k << "const ulong numWorkers = get_local_size(0);\n";
	//ensure we are not reading out of bounds
	k << "const ulong curTileA = end-start;\n";
	
	// Load tile of A into local memory
	k << "for(ulong i = get_local_id(0); i < TILE_SIZE; i += numWorkers){\n";
	k << "	for(ulong j = 0; j < TILE_SIZE; j++){\n";
	k << "		Asub[i][j] ="<< A(k.expr<cl_ulong>("min(end-1, start + i)"),k.expr<cl_ulong>("min(end-1, start + j)"))<<";\n";
	k << "	}\n";
	k << "}\n";
	
	
	// Load Tile of B into local memory, store columns of B as rows
	k << "for(ulong i = get_local_id(0); i < TILE_SIZE; i += numWorkers){\n";
	k << "	Bsub[i] ="<< b(k.expr<cl_ulong>("min(end-1,start + i)"))<<";\n";
	k << "}\n";
	// Synchronise to make sure everything is loaded
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";

	// Loop over the values of a single tile
	//lower-case
	k << "if(!upper){\n";
	k << "	for(ulong i = 0; i < TILE_SIZE && get_local_id(0) == 0; ++i){\n";
	k << "		if(!unit){Bsub[i] /= Asub[i][i];}\n";
	k << "		for(ulong j = i+1; j < TILE_SIZE; ++j){\n";
	k << "			Bsub[j] -= "<< prod(k.expr<value_typeB>("Bsub[i]"), k.expr<value_typeA>("Asub[j][i]"))<<";\n";
	k << "		}\n";
	k << "	}\n";
	k << "}else{\n";
	//upper case
	k << "	for(ulong n = curTileA; n > 0 && get_local_id(0) == 0; --n){\n";
	k << "		ulong i = n-1;\n";
	k << "		if(!unit ){Bsub[i] /= Asub[i][i];}\n";
	k << "		for(ulong j = 0; j < i; j ++){\n";
	k << "			Bsub[j] -= "<< prod(k.expr<value_typeB>("Bsub[i]"), k.expr<value_typeA>("Asub[j][i]"))<<";\n";
	k << "		}\n";
	k << "	}\n";
	k << "}\n";
	// Synchronise before continuing
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";
	// Store the final results back in B
	k << "for(ulong i = get_local_id(0); i < curTileA; i += numWorkers){\n";
	k << b(k.expr<cl_ulong>("(start+i)"))<<" =  Bsub[i];\n";
	k << "}\n";
	
	boost::compute::kernel kernel = k.compile(b_unreg().queue().get_context(), options);
	return {kernel,start_index,end_index,unit_index,upper_index};
}

template <typename MatA, typename VecB, class Triangular>
void trsv_recursive(
	matrix_expression<MatA, gpu_tag> const& Afull, 
	vector_expression<VecB, gpu_tag> & bfull,
	trsv_kernel& kernel,
	std::size_t start,
	std::size_t end,
	std::size_t tileSize,
	std::size_t numWorkers,
	Triangular t
){
	
	std::size_t size = end-start;
	//if the matrix is small enough call the computation kernel directly for the block
	if(size <= tileSize){
		//enqueue kernel with kernel args
		kernel.kernel.set_arg(kernel.start_index, start);
		kernel.kernel.set_arg(kernel.end_index, end);
		kernel.kernel.set_arg(kernel.unit_index, (std::size_t)Triangular::is_unit);
		kernel.kernel.set_arg(kernel.upper_index, (std::size_t)Triangular::is_upper);
		
		std::size_t global_work_size[2] = {numWorkers,1};
		std::size_t local_work_size[2] = {numWorkers, 1};
		bfull().queue().enqueue_nd_range_kernel(kernel.kernel, 2,nullptr, global_work_size, local_work_size);
		return;
	}
	std::size_t numBlocks = (size+tileSize-1)/tileSize;
	std::size_t split = numBlocks/2 * tileSize;
	auto bfront = subrange(bfull,start,start+split);
	auto bback = subrange(bfull,start+split,end);
	
	//otherwise run the kernel recursively
	if(Triangular::is_upper){ //Upper triangular case
		auto Aur = subrange(Afull,start,start+split,start+split,end);
		trsv_recursive(Afull, bfull, kernel, start+split,end, tileSize, numWorkers, t);
		kernels::gemv(Aur, bback, bfront, -1.0);
		trsv_recursive(Afull, bfull, kernel, start,start+split, tileSize, numWorkers, t);
	}else{// Lower triangular caste
		auto All = subrange(Afull,start+split,end,start,start+split);
		trsv_recursive(Afull, bfull, kernel, start,start+split, tileSize, numWorkers, t);
		kernels::gemv(All, bfront, bback, -1.0);
		trsv_recursive(Afull, bfull, kernel, start+split,end, tileSize, numWorkers, t);
	}
}

template <typename MatA, typename VecB, class Triangular>
void trsv_call(
	matrix_expression<MatA, gpu_tag> const& A, 
	vector_expression<VecB, gpu_tag>& b,
	Triangular,
	left
){
	std::size_t const TileSize = 32;//size of the diagonal blocks where the single kernel runs
	std::size_t const numWorkers = TileSize; //number of workers
	char const* options ="-DTILE_SIZE=32ul";
	auto kernel = bindings::createTRSVDiagBlockKernel(A,b,options);
	trsv_recursive(A,b,kernel,0,A().size1(), TileSize, numWorkers, Triangular());
}

template <typename MatA, typename VecB, class Triangular>
void trsv_call(
	matrix_expression<MatA, gpu_tag> const& A, 
	vector_expression<VecB, gpu_tag>& b,
	Triangular,
	right
){
	trsv_call(trans(A),b,typename Triangular::transposed_orientation(),left());
}
}
namespace kernels{
//main kernel runs the kernel above recursively and calls gemv
template <class Triangular,class Side, typename MatA, typename VecB>
void trsv(
	matrix_expression<MatA, gpu_tag> const& A, 
	vector_expression<VecB, gpu_tag>& b
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size2() == b().size());
	bindings::trsv_call(A,b,Triangular(), Side());
}
}}
#endif
