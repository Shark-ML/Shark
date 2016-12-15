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

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <boost/compute/kernel.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/functional/operator.hpp> //for multiplies

namespace shark { namespace blas {namespace bindings{

template<class M,class V, class Orientation>
void sum_rows(
	matrix_expression<M, gpu_tag> const& A, 
	vector_expression<V, gpu_tag>& v,
	typename V::value_type alpha,
	Orientation, dense_tag, dense_tag
){
	typedef typename V::value_type value_type;
	boost::compute::detail::meta_kernel k("blas_sum_rows_row");
	std::size_t alpha_index = k.add_arg<value_type>("alpha");
	std::size_t size1_index = k.add_arg<std::size_t>("size1");
	std::size_t size2_index = k.add_arg<std::size_t>("size2");
	//read all tiles in the assigned rows and sum them up
	k << "__local " <<k.decl<value_type>("sums")<< "[TILE_DIM][TILE_DIM+1];\n";
	k << "uint colid = get_global_id(1);\n";
	k << "sums[get_local_id(0)][get_local_id(1)]  = 0.0;\n";
	k << "for(uint i = get_local_id(0) ; i < size1 && colid < size2; i += TILE_DIM){\n";
	auto exprRow = k.expr<cl_uint>("i");
	auto exprCol = k.expr<cl_uint>("colid");
	k<< "    sums[get_local_id(0)][get_local_id(1)] +=" << A()(exprRow,exprCol)<<";\n";
	k<<'}';
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";//wait until all threads are done with copying
	//sum up the rows
	k << "if(get_local_id(0) == 0 && colid < size2){\n";
	k << "    for(uint i = 1 ; i < TILE_DIM; ++i){\n";
	k << "        sums[0][get_local_id(1)] +=sums[i][get_local_id(1)];\n";
	k << "    }\n";
	k << v()(exprCol) << "+= alpha * sums[0][get_local_id(1)];\n";
	k<< "}\n";
	//create source

	std::size_t TILE_DIM = 8;
	char const* options ="-DTILE_DIM=8";
	boost::compute::kernel kernel = k.compile(v().queue().get_context(), options);
	//enqueue kernel
	kernel.set_arg(alpha_index, alpha);
	kernel.set_arg(size1_index, A().size1());
	kernel.set_arg(size2_index, A().size2());
	
	std::size_t global_work_size[2] = {
		TILE_DIM,
		((A().size2()+TILE_DIM-1)/TILE_DIM) * TILE_DIM
	};
	std::size_t local_work_size[2] = {TILE_DIM, TILE_DIM};
	v().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, local_work_size);
}


}}}

#endif
