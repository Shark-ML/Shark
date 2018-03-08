/*!
 *
 *
 * \brief       Implements the default implementation of the POTRF algorithm
 *
 * \author    O. Krause
 * \date        2016
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
#ifndef REMORA_KERNELS_GPU_POTRF_HPP
#define REMORA_KERNELS_GPU_POTRF_HPP

#include "../../proxy_expressions.hpp"
#include "../trsm.hpp" //trsm kernel
#include "../syrk.hpp" //syrk kernel

namespace remora{namespace bindings {
	
	
struct potrf_kernel{
	boost::compute::kernel kernel;
	std::size_t start_index;
	std::size_t end_index;
};
//Lower triangular - matrix(row-major)
template<class MatA>
potrf_kernel createPotrfDiagBlockKernel(
	matrix_expression<MatA, gpu_tag>& A_unreg,
	char const* options
){
	typedef typename MatA::value_type value_type;
	
	gpu::detail::meta_kernel k("blas_potrf");
	std::size_t start_index = k.add_arg<std::size_t>("start");//start of block of A
	std::size_t end_index = k.add_arg<std::size_t>("end");//end of Block of A
	auto A = k.register_args(to_functor(A_unreg));
	// Local memory to fit a tile of A and B
	// we store B as column major in local memory
	// we also allocate memory to store results of B
	k << "__local " <<k.decl<value_type>("Asub")<< "[TILE_SIZE][TILE_SIZE+2];\n";//+2 to avoid bank conflicts
	k << "const ulong numWorkers = get_local_size(0);\n";
	//ensure we are not reading out of bounds
	k << "const ulong t = get_group_id(1);\n";
	k << "const ulong curTileA = end-start;\n";
	
	// Load tile of A into local memory
	k << "for(ulong i = get_local_id(0); i < TILE_SIZE; i += numWorkers){\n";
	k << "	for(ulong j = get_local_id(1); j < TILE_SIZE; j += numWorkers){\n";
	k << "		Asub[i][j] ="<< A(k.expr<cl_ulong>("min(end-1, start + i)"),k.expr<cl_ulong>("min(end-1, start + j)"))<<";\n";
	k << "	}\n";
	k << "}\n";
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";
	// Loop over the values of a single tile
	//upper-case
	k << "if(get_local_id(0) == 0 && get_local_id(1) == 0){\n";
	k << "    for(ulong j = 0; j < TILE_SIZE; j++) {\n";
	k << k.decl<value_type>("Ajj") <<"= sqrt(Asub[j][j]);\n";
	k << "        Asub[j][j] = Ajj;\n";
	k << "        for(ulong i = j + 1; i < TILE_SIZE; ++i) {\n";
	k << "            Asub[i][j] /= Ajj;\n";
	k << "        }\n";
		//rank-one update
	k << "        for(ulong k = j + 1; k < TILE_SIZE; k++) {\n";
	k << "            for(ulong i = k; i < TILE_SIZE; ++i) {\n";
	k << "                 Asub[i][k] -= Asub[i][j] * Asub[k][j];\n";
	k << "            }\n";
	k << "        }\n";
	k << "    }\n";
	k << "}\n";
	// Synchronise before continuing
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";
	// Store the final results back in A
	k << "for(ulong i = get_local_id(0); i < curTileA; i += numWorkers){\n";
	k << "	for(ulong j = get_local_id(1); j < curTileA; j += numWorkers){\n";
	k << A(k.expr<cl_ulong>("(start+i)"),k.expr<cl_ulong>("(start+j)"))<<" = Asub[i][j];\n";
	k << "	}\n";
	k << "}\n";
	
	boost::compute::kernel kernel = k.compile(A_unreg().queue().get_context(), options);
	return {kernel,start_index,end_index};
}
//main kernel for large matrices
template <typename MatA>
void potrf_recursive(
	matrix_expression<MatA, gpu_tag>& Afull,
	std::size_t start,
	std::size_t end,
	potrf_kernel& kernel
){
	std::size_t block_size = 16;
	std::size_t num_workers = 8;
	auto A = subrange(Afull,start,end,start,end);
	std::size_t size = A.size1();
	//if the matrix is small enough call the computation kernel directly for the block
	if(size <= block_size){
		kernel.kernel.set_arg(kernel.start_index, start);
		kernel.kernel.set_arg(kernel.end_index, end);
		
		std::size_t global_work_size[2] = {num_workers,num_workers};
		std::size_t local_work_size[2] = {num_workers, num_workers};
		Afull().queue().enqueue_nd_range_kernel(kernel.kernel, 2,nullptr, global_work_size, local_work_size);
		return;
	}
	std::size_t numBlocks = (A.size1()+block_size-1)/block_size;
	std::size_t split = numBlocks/2*block_size;
	
	
	//otherwise run the kernel recursively
	potrf_recursive(Afull,start,start+split, kernel);
	
	auto Aul = subrange(A,0,split,0,split);
	auto All = subrange(A,split,size,0,split);
	auto Alr = subrange(A,split,size,split,size);
	kernels::trsm<upper,right>(trans(Aul), All );
	kernels::syrk<false>(All,Alr, -1.0);
	potrf_recursive(Afull,start+split,end, kernel);
}

template <typename MatA>
void potrf_dispatch(
	matrix_expression<MatA, gpu_tag>& A,
	lower
){
	char const* options ="-DTILE_SIZE=16ul";
	auto kernel = bindings::createPotrfDiagBlockKernel(A,options);
	potrf_recursive(A,0,A().size1(), kernel);
}
template <typename MatA>
void potrf_dispatch(
	matrix_expression<MatA, gpu_tag>& A,
	upper
){
	auto Atrans = trans(A);
	char const* options ="-DTILE_SIZE=16ul";
	auto kernel = bindings::createPotrfDiagBlockKernel(Atrans,options);
	potrf_recursive(Atrans,0,Atrans().size1(), kernel);
}
}

namespace kernels {
template <class Triangular, typename MatA>
std::size_t potrf(
	matrix_container<MatA, gpu_tag>& A
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	bindings::potrf_dispatch(A, Triangular());
	return 0;
}

}}
#endif
