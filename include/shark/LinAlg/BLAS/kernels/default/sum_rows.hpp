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

#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_SUM_ROWS_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_SUM_ROWS_HPP

#include "../../assignment.hpp"

namespace shark { namespace blas {namespace bindings{
	
template<class M,class V, class Tag1, class Tag2>
void sum_rows(
	matrix_expression<M, cpu_tag> const& A, 
	vector_expression<V, cpu_tag>& v,
	typename V::value_type alpha,
	column_major, Tag1, Tag2
){
	for(std::size_t i = 0; i != A().size2(); ++i){
		typename V::value_type sum = 0;
		auto end = A().column_end(i);
		for(auto pos = A().column_begin(i); pos != end; ++pos)
			sum += *pos;
		v()(i) += alpha * sum;
	}
}

template<class M,class V, class Tag1, class Tag2>
void sum_rows(
	matrix_expression<M, cpu_tag> const& A, 
	vector_expression<V, cpu_tag>& v,
	typename V::value_type alpha,
	row_major, Tag1, Tag2
){
	for(std::size_t i = 0; i != A().size1(); ++i){
		auto end = A().row_end(i);
		for(auto pos = A().row_begin(i); pos != end; ++pos)
			v()(pos.index()) += alpha * (*pos);
	}
}



//dispatcher for triangular matrix
template<class M,class V,class Orientation,class Triangular, class Tag1, class Tag2>
void sum_rows(
	matrix_expression<M, cpu_tag> const& A, 
	vector_expression<V, cpu_tag>& v,
	typename V::value_type alpha,
	triangular<Orientation,Triangular>, Tag1, Tag2
){
	sum_rows(A,v,alpha,Orientation(), Tag1(), Tag2());
}

}}}

#endif
