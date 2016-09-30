/*!
 * 
 *
 * \brief       Sums the rows of a row-major or column major matrix.
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

#ifndef SHARK_LINALG_BLAS_KERNELS_CLBLAS_SUM_ROWS_HPP
#define SHARK_LINALG_BLAS_KERNELS_CLBLAS_SUM_ROWS_HPP

#include "../../detail/traits.hpp"
#include <boost/compute/detail/meta_kernel.hpp>

namespace shark { namespace blas {namespace bindings{

template<class M,class V, class Orientation>
void sum_rows(
	matrix_expression<M, gpu_tag> const& A, 
	vector_expression<V, gpu_tag>& v,
	typename V::value_type alpha,
	Orientation, dense_tag, dense_tag
){
	typedef typename M::value_type value_type;
	boost::compute::detail::meta_kernel k("blas_sum_rows_row");
	std::size_t alpha_index = k.add_arg<value_type>("alpha");
	std::size_t size1_index = k.add_arg<std::size_t>("size1");
	std::size_t size2_index = k.add_arg<std::size_t>("size2");
	//read all tiles in the assigned rows and sum them up
	k << "__local" <<k.decl<value_type>("sums")<< "[TILE_DIM][TILE_DIM+1];";
	k << " uint rowid = get_group_id(0) * TILE_DIM + get_local_id(0);";
	k << "for(uint i = get_local_id(1) ; i < size2 && rowid < size1; i += TILE_DIM){";
	auto exprRow = k.expr<cl_uint>("rowid");
	auto exprCol = k.expr<cl_uint>("get_group_id(0) * TILE_DIM + i");
	k<< "    sums[get_local_id(0)][get_local_id(1)] +=" << A()(exprRow,exprCol)<<";";
	k<<'}';
	k << "barrier(CLK_LOCAL_MEM_FENCE);";//wait until all threads are done with copying
	//sum up the columns
	k << "if(get_local_id(1) == 0){";
	k << "    for(uint i = 1 ; i < TILE_DIM; ++i){";
	k << "        sums[get_local_id(0)][0] +=sums[get_local_id(0)][i];";
	k << "    }";
	k << v()(exprRow) << "= alpha * sums[get_local_id(0)][0];";
	k<< "}";
	//create source

	std::size_t TILE_DIM = 8;
	char const* options ="-DTILE_DIM = 8";
	boost::compute::kernel kernel = k.compile(v().queue().get_context(), options);
	//enqueue kernel
	kernel.set_arg(alpha_index, alpha);
	kernel.set_arg(size1_index, A().size1());
	kernel.set_arg(size2_index, A().size2());
	
	std::size_t global_work_size[2] = {
		(A().size1()+TILE_DIM-1)/TILE_DIM,
		(A().size2()+TILE_DIM-1)/TILE_DIM,
	};
	std::size_t local_work_size[2] = {TILE_DIM, TILE_DIM};
	v().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, local_work_size);
}


}}}

#endif
