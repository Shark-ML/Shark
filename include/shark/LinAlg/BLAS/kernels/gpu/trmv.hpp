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
#ifndef REMORA_KERNELS_CLBLAS_TRMV_HPP
#define REMORA_KERNELS_CLBLAS_TRMV_HPP


#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <boost/compute/functional/operator.hpp> //for multiplies
#include "../gemv.hpp"

namespace remora {namespace bindings {

struct trmv_kernel{
	boost::compute::kernel kernel;
	std::size_t start_index;
	std::size_t end_index;
	std::size_t unit_index;
	std::size_t upper_index;
};
//Lower triangular - matrix(row-major)
template<class MatA, class VecV>
trmv_kernel createTRMVBlockKernel(
	matrix_expression<MatA, gpu_tag> const& A_unreg,
	vector_expression<VecV, gpu_tag>& v_unreg,
	char const* options
){
	typedef typename MatA::value_type value_typeA;
	typedef typename VecV::value_type value_typeV;
	boost::compute::multiplies<value_typeV> prod;
	
	gpu::detail::meta_kernel k("blas_trmv");
	std::size_t start_index = k.add_arg<std::size_t>("start");//start of block of A
	std::size_t end_index = k.add_arg<std::size_t>("end");//end of Block of A
	std::size_t unit_index = k.add_arg<std::size_t>("unit");//whether A is unit triangular
	std::size_t upper_index = k.add_arg<std::size_t>("upper");//whether A is unit triangular
	auto A = k.register_args(to_functor(A_unreg));
	auto v = k.register_args(to_functor(v_unreg));
	// Local memory to fit a tile of A and B
	// we store B as column major in local memory
	// we also allocate memory to store results of B
	k << "__local " <<k.decl<value_typeA>("Asub")<< "[TILE_SIZE][TILE_SIZE+2];\n";//+2 to avoid bank conflicts
	k << "__local " <<k.decl<value_typeV>("Bsub")<< "[TILE_SIZE];\n";
	k << "__local " <<k.decl<value_typeV>("BResult")<< "[TILE_SIZE];\n";
	k << "	const ulong numWorkers = get_local_size(0);\n";
	
	// Load tile of A into local memory
	k << "const ulong curTileA =  end-start;\n";
	k << "for(ulong i = 0; i < curTileA; ++i){\n";
	k << "	for(ulong j = get_local_id(0); j < curTileA; j += numWorkers){\n";
	k << "		Asub[i][j] ="<< A(k.expr<cl_ulong>("(i+start)"),k.expr<cl_ulong>("(j+start)"))<<";\n";
	k << "	}\n";
	k << "}\n";
		
	//ensure we are not reading out of bounds
	// Load Tile of B into local memory, store columns of B as rows
	k << "for(ulong i = get_local_id(0); i < curTileA; i += numWorkers){\n";
	k << "	Bsub[i] = "<< v(k.expr<cl_ulong>("(start+i)"))<<";\n";
	k << "}\n";
	// Synchronise to make sure the tile is loaded
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";

	// Loop over the values of a single tile
	// by computing outer products ulongo the local accumulation registers acc
	//lower-case
	k << "if(!upper){\n";
	k << "	for(ulong i = get_local_id(0); i < TILE_SIZE; i += numWorkers){\n";
	k << "		BResult[i] = Bsub[i];\n";
	k << "		if(!unit){BResult[i] *= Asub[i][i];}\n";
	k << "		for(ulong j = 0; j < i; ++j){\n";
	k << "			BResult[i] +="<< prod(k.expr<value_typeV>("Bsub[j]"), k.expr<value_typeA>("Asub[i][j]"))<<";\n";
	k << "		}\n";
	k << "	}\n";
	k << "}else{\n";
	//upper case
	k << "	for(ulong i = get_local_id(0); i < curTileA; i += numWorkers){\n";
	k << "		BResult[i] = Bsub[i];\n";
	k << "		if(!unit){BResult[i] *= Asub[i][i];}\n";
	k << "			for(ulong j = i+1; j < curTileA; ++j){\n";
	k << "				BResult[i] +="<< prod(k.expr<value_typeV>("Bsub[j]"), k.expr<value_typeA>("Asub[i][j]"))<<";\n";
	k << "		}\n";
	k << "	}\n";
	k << "}\n";
	// Synchronise before loading the next tile
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";
	// Store the final results back in B
	k << "for(ulong i = get_local_id(0); i < curTileA; i += numWorkers){\n";
	k << v(k.expr<cl_ulong>("(start+i)"))<<" =  BResult[i];\n";
	k << "}\n";
	
	boost::compute::kernel kernel = k.compile(v_unreg().queue().get_context(), options);
	return {kernel,start_index,end_index,unit_index,upper_index};
}

template <typename MatA, typename VecV, typename Triangular>
void trmv_recursive(
	matrix_expression<MatA, gpu_tag> const& Afull, 
	vector_expression<VecV, gpu_tag> & vfull,
	trmv_kernel& kernel,
	std::size_t start,
	std::size_t end,
	std::size_t tileSizeA,
	Triangular t
){
	std::size_t size = end-start;
	
	//if the matrix is small enough, call the computation kernel directly for the block
	if(size <= tileSizeA){
		//~ std::cout<<"called "<<size<<" "<<start<<" "<<end<<" "<<Afull().raw_storage().leading_dimension<<std::endl;
	
		//enqueue kernel with kernel args
		kernel.kernel.set_arg(kernel.start_index, start);
		kernel.kernel.set_arg(kernel.end_index, end);
		kernel.kernel.set_arg(kernel.unit_index, (std::size_t)Triangular::is_unit);
		kernel.kernel.set_arg(kernel.upper_index, (std::size_t)Triangular::is_upper);
		
		std::size_t global_work_size[2] = {
			tileSizeA,
			1
		};
		vfull().queue().enqueue_nd_range_kernel(kernel.kernel, 2,nullptr, global_work_size, global_work_size);
		return;
	}
	//otherwise run the kernel recursively
	std::size_t split = ((size+tileSizeA-1)/tileSizeA)/2 * tileSizeA;//split at the next multiple of the TileSize
	auto vfront = subrange(vfull,start,start+split);
	auto vback = subrange(vfull,start+split,end);
	
	if(Triangular::is_upper){ //Upper triangular case
		auto Aur = subrange(Afull,start,start+split,start+split,end);
		trmv_recursive(Afull, vfull, kernel, start, start+split, tileSizeA, t);
		kernels::gemv(Aur, vback, vfront, 1.0);
		trmv_recursive(Afull, vfull, kernel, start+split, end, tileSizeA, t);
	}else{// Lower triangular caste
		auto All = subrange(Afull,start+split,end,start,start+split);
		trmv_recursive(Afull, vfull, kernel, start+split, end, tileSizeA, t);
		kernels::gemv(All, vfront, vback, 1.0);
		trmv_recursive(Afull, vfull, kernel, start, start+split, tileSizeA, t);
	}

}
}
namespace kernels{
//main kernel runs the kernel above recursively and calls gemv
template <bool Upper,bool Unit,typename MatA, typename VecV>
void trmv(
	matrix_expression<MatA, gpu_tag> const& A, 
	vector_expression<VecV, gpu_tag>& v
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size2() == v().size());
	
	std::size_t const TileSizeA = 32;//size of the diagonal blocks where the single kernel runs
	char const* options ="-DTILE_SIZE=32ul";
	auto kernel = bindings::createTRMVBlockKernel(A,v,options);
	
	bindings::trmv_recursive(A,v,kernel,0,A().size1(), TileSizeA, triangular_tag<Upper,Unit>());

}

}}
#endif
