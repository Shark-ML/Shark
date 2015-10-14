/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2011
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
#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_TRSV_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_TRSV_HPP

#include "../../matrix_proxy.hpp"
#include "../../vector_expression.hpp"
#include <boost/mpl/bool.hpp>

namespace shark {namespace blas {namespace bindings {

//tag encoding
// Upper matrix -> boost::mpl::true_
// Lower matrix -> boost::mpl::false_
	
//Lower triangular(row-major) - vector
template<bool Unit, class TriangularA, class V>
void trsv_impl(
	matrix_expression<TriangularA> const& A,
	vector_expression<V> &b,
        boost::mpl::false_, column_major
) {
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == b().size());
	
	typedef typename TriangularA::value_type value_type;
	
	std::size_t size = b().size();
	for (std::size_t n = 0; n != size; ++ n) {
		if(!Unit){
			RANGE_CHECK(A()(n, n) != value_type());//matrix is singular
			b()(n) /= A()(n, n);
		}
		if (b()(n) != value_type/*zero*/()){
			matrix_column<TriangularA const> col = column(A(),n);
			noalias(subrange(b(),n+1,size)) -= b()(n) * subrange(col,n+1,size);
		}
	}
}
//Lower triangular(column-major) - vector
template<bool Unit, class TriangularA, class V>
void trsv_impl(
	matrix_expression<TriangularA> const& A,
	vector_expression<V> &b,
        boost::mpl::false_, row_major
) {
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == b().size());
	
	typedef typename TriangularA::value_type value_type;
	
	std::size_t size = b().size();
	for (std::size_t n = 0; n < size; ++ n) {
		matrix_row<TriangularA const> matRow = row(A(),n);
		b()(n) -= inner_prod(subrange(matRow,0,n),subrange(b(),0,n));

		if(!Unit){
			RANGE_CHECK(A()(n, n) != value_type());//matrix is singular
			b()(n) /= A()(n, n);
		}
	}
}

//upper triangular(column-major)-vector
template<bool Unit, class TriangularA, class V>
void trsv_impl(
	matrix_expression<TriangularA> const& A,
	vector_expression<V> &b,
        boost::mpl::true_, column_major
) {
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == b().size());
	
	typedef typename TriangularA::value_type value_type;
	
	std::size_t size = b().size();
	for (std::size_t i = 0; i < size; ++ i) {
		std::size_t n = size-i-1;
		if(!Unit){
			RANGE_CHECK(A()(n, n) != value_type());//matrix is singular
			b()(n) /= A()(n, n);
		}
		if (b()(n) != value_type/*zero*/()) {
			matrix_column<TriangularA const> col = column(A(),n);
			noalias(subrange(b(),0,n)) -= b()(n) * subrange(col,0,n);
		}
	}
}
//upper triangular(row-major)-vector
template<bool Unit, class TriangularA, class V>
void trsv_impl(
	matrix_expression<TriangularA> const& A,
	vector_expression<V> &b,
        boost::mpl::true_, row_major
) {
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == b().size());
	
	typedef typename TriangularA::value_type value_type;
	
	std::size_t size = A().size1();
	for (std::size_t i = 0; i < size; ++ i) {
		std::size_t n = size-i-1;
		matrix_row<TriangularA const> matRow = row(A(),n);
		b()(n) -= inner_prod(subrange(matRow,n+1,size),subrange(b(),n+1,size));
		if(!Unit){
			RANGE_CHECK(A()(n, n) != value_type());//matrix is singular
			b()(n) /= A()(n, n);
		}
	}
}

//dispatcher

template <bool Upper,bool Unit,typename TriangularA, typename V>
void trsv(
	matrix_expression<TriangularA> const& A, 
	vector_expression<V> & b,
	boost::mpl::false_//unoptimized
){
	trsv_impl<Unit>(A, b, boost::mpl::bool_<Upper>(), typename TriangularA::orientation());
}

}}}
#endif
