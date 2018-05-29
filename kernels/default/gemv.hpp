/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2012
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
 * MatAERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef REMORA_KERNELS_DEFAULT_GEMatAV_HPP
#define REMORA_KERNELS_DEFAULT_GEMatAV_HPP

#include "../../expression_types.hpp" //matrix/vector_expression
#include "../../proxy_expressions.hpp" //matrix row,, transpose
#include "../../detail/traits.hpp" //matrix orientations
#include "../dot.hpp" //inner product
#include "../vector_assign.hpp" //assignment of vectors
#include <type_traits> //std::false_type marker for unoptimized

namespace remora{namespace bindings {
	
//row major can be further reduced to inner_prod()
template<class ResultV, class MatA, class V>
void gemv_impl(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag> const& x,
	vector_expression<ResultV, cpu_tag>& result, 
	typename ResultV::value_type alpha,
	row_major
) {
	typedef typename ResultV::value_type value_type;
	value_type value;
	for(std::size_t i = 0; i != A().size1();++i){
		kernels::dot(row(A,i),x,value);
		if(value != value_type())//handling of sparse results.
			result()(i) += alpha * value;
	}
}

//column major is implemented by computing a linear combination of matrix-rows 
template<class ResultV, class MatA, class V>
void gemv_impl(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag> const& x,
	vector_expression<ResultV, cpu_tag>& result,
	typename ResultV::value_type alpha,
	column_major
) {
	typedef typename V::const_iterator iterator;
	typedef typename ResultV::value_type value_type;
	typedef device_traits<cpu_tag>::multiply_and_add<value_type> MultAdd;
	iterator end = x().end();
	for(iterator it = x().begin(); it != end; ++it) {
		//FIXME: for sparse result vectors, this might hurt.
		kernels::assign(result, column(A,it.index()), MultAdd(alpha * (*it)));
	}
}

//unknown orientation is dispatched to row_major
template<class ResultV, class MatA, class V>
void gemv_impl(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag> const& x,
	vector_expression<ResultV, cpu_tag>& result,
	typename ResultV::value_type alpha,
	unknown_orientation
) {
	gemv_impl(A,x,result,alpha,row_major());
}

// result += alpha * A * x
template<class ResultV, class MatA, class V>
void gemv(
	matrix_expression<MatA, cpu_tag> const& A,
        vector_expression<V, cpu_tag> const& x,
        vector_expression<ResultV, cpu_tag>& result, 
	typename ResultV::value_type alpha,
	std::false_type
) {
	typedef typename MatA::orientation orientation;

	gemv_impl(A, x, result, alpha, orientation());
}

}}
#endif
