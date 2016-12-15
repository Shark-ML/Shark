/*!
 * \brief       kernels for folding matrices with openCL
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
#ifndef SHARK_LINALG_BLAS_KERNELS_CLBLAS_MATRIX_FOLD_HPP
#define SHARK_LINALG_BLAS_KERNELS_CLBLAS_MATRIX_FOLD_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <boost/compute/kernel.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/container/array.hpp>
#include <boost/compute/algorithm/copy_n.hpp>
namespace shark{namespace blas {namespace bindings{

template<class F, class MatA, class Orientation>
void matrix_fold(matrix_expression<MatA, gpu_tag> const& A, typename F::result_type& value, Orientation, dense_tag) {
	auto& queue = A().queue();
	typedef typename F::result_type value_type;
	boost::compute::detail::meta_kernel k("blas_matrix_fold");
	std::size_t size1_index = k.add_arg<std::size_t>("size1");
	std::size_t size2_index = k.add_arg<std::size_t>("size2");
	boost::compute::array<value_type,1> device_result;
	boost::compute::copy_n(&value, 1, device_result.begin(), queue);
	device_result.front() = value;
	F f;
	
	//read all tiles in the assigned rows and apply f
	k << "__local " <<k.decl<value_type>("subfold")<< "[TILE_DIM][TILE_DIM+1];";
	k << "subfold[get_local_id(0)][get_local_id(1)]  = "<<device_result.begin()[0]<<';';
	k << "for(uint i = get_local_id(0) ; i < size1; i += TILE_DIM){";
	k << "    for(uint j = get_local_id(1) ; j < size2; j += TILE_DIM){";
	auto exprSubFold = k.expr<value_type>("subfold[get_local_id(0)][get_local_id(1)]");
	k<< exprSubFold << '=' << f(exprSubFold,A()(k.expr<cl_uint>("i"),k.expr<cl_uint>("j")))<<";";
	k<<"}}";
	k << "barrier(CLK_LOCAL_MEM_FENCE);";//wait until all threads are done with copying
	//sum up the rows
	k << "if(get_local_id(0) == 0){";
	k << "    for(uint i = 1 ; i < TILE_DIM; ++i){";
	k << "        subfold[0][get_local_id(1)] =" 
		<< f(
			k.expr<value_type>("subfold[0][get_local_id(1)]"),
			k.expr<value_type>("subfold[i][get_local_id(1)]")
		)<<';';
	k << "    }";
	k <<"    if(get_local_id(1) == 0){";
	k << "       for(uint i = 1 ; i < TILE_DIM; ++i){";
	k <<"            subfold[0][0] =" << f(k.expr<value_type>("subfold[0][0]"),k.expr<value_type>("subfold[0][i]"))<<';';
	k <<"        }";
	k <<device_result.begin()[0]<< "= subfold[0][0];";
	k<< "}}";
	
	//compile kernel
	std::size_t TILE_DIM = 1;
	char const* options ="-DTILE_DIM=1";
	boost::compute::kernel kernel = k.compile(queue.get_context(), options);
	//enqueue kernel
	kernel.set_arg(size1_index, A().size1());
	kernel.set_arg(size2_index, A().size2());
	
	std::size_t global_work_size[2] = {TILE_DIM,TILE_DIM};
	std::size_t local_work_size[2] = {TILE_DIM, TILE_DIM};
	queue.enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, local_work_size);
	boost::compute::copy_n(device_result.begin(), 1, &value, queue);
}
}}}
#endif
