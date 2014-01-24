/*!
 * 
 * \file        default/trsm.hpp
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2012
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

#ifndef SHARK_LINALG_BLAS_UBLAS_KERNELS_ATLAS_TRSM_HPP
#define SHARK_LINALG_BLAS_UBLAS_KERNELS_ATLAS_TRSM_HPP

#include "../../matrix_proxy.hpp"
#include "../../vector_expression.hpp"
#include <boost/mpl/bool.hpp>

namespace shark {namespace blas {namespace bindings {
	
//fixme: no handling of orientation of MatB. Might be solved by rewriting with block-algorithms.

// Lower triangular(column major) - matrix
template<bool Unit, class MatA, class MatB>
void trsm_impl(
	matrix_expression<MatA> const& A, matrix_expression<MatB>& B, 
	boost::mpl::false_, column_major
) {
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == B().size1());
	
	typedef typename MatA::value_type value_type;
	
	std::size_t size1 = B().size1();
	std::size_t size2 = B().size2();
	for (std::size_t n = 0; n < size1; ++ n) {
		matrix_column<MatA const> columnTriangular = column(A(),n);
		for (std::size_t l = 0; l < size2; ++ l) {
			if(!Unit){
				RANGE_CHECK(A()(n, n) != value_type());//matrix is singular
				B()(n, l) /= A()(n, n);
			}
			if (B()(n, l) != value_type/*zero*/()) {
				matrix_column<MatB> columnMatrix = column(B(),l);
				noalias(subrange(columnMatrix,n+1,size1)) -= B()(n,l) * subrange(columnTriangular,n+1,size1);
			}
		}
	}
}
// Lower triangular(row major) - matrix
template<bool Unit, class MatA, class MatB>
void trsm_impl(
	matrix_expression<MatA> const& A, matrix_expression<MatB>& B, 
	boost::mpl::false_, row_major
) {
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == B().size1());
	
	typedef typename MatA::value_type value_type;
	
	std::size_t size1 = B().size1();
	for (std::size_t n = 0; n < size1; ++ n) {
		for (std::size_t m = 0; m < n; ++m) {
			noalias(row(B(),n)) -= A()(n,m)*row(B(),m);
		}
		if(!Unit){
			RANGE_CHECK(A()(n, n) != value_type());//matrix is singular
			row(B(),n)/=A()(n, n);
		}
	}
}

//Upper triangular(column major) - matrix
template<bool Unit, class MatA, class MatB>
void trsm_impl(
	matrix_expression<MatA> const& A, matrix_expression<MatB>& B,
        boost::mpl::true_, column_major
) {
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == B().size1());
	
	typedef typename MatA::value_type value_type;
	
	std::size_t size1 = B().size1();
	std::size_t size2 = B().size2();
	for (std::size_t i = 0; i < size1; ++ i) {
		std::size_t n = size1-i-1;
		matrix_column<MatA const> columnTriangular = column(A(),n);
		if(!Unit){
			RANGE_CHECK(A()(n, n) != value_type());//matrix is singular
			row(B(),n) /= A()(n, n);
		}
		for (std::size_t l = 0; l < size2; ++ l) {
			if (B()(n, l) != value_type/*zero*/()) {
				matrix_column<MatB> columnMatrix = column(B(),l);
				noalias(subrange(columnMatrix,0,n)) -= B()(n,l) * subrange(columnTriangular,0,n);
			}
		}
	}
}

//Upper triangular(row major) - matrix
template<bool Unit, class MatA, class MatB>
void trsm_impl(
	matrix_expression<MatA> const& A, matrix_expression<MatB>& B,
        boost::mpl::true_, row_major
) {
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == B().size1());
	
	typedef typename MatA::value_type value_type;
	
	std::size_t size1 = B().size1();
	for (std::size_t i = 0; i < size1; ++ i) {
		std::size_t n = size1-i-1;
		for (std::size_t m = n+1; m < size1; ++m) {
			noalias(row(B(),n)) -= A()(n,m)*row(B(),m);
		}
		if(!Unit){
			RANGE_CHECK(A()(n, n) != value_type());//matrix is singular
			row(B(),n)/=A()(n, n);
		}
	}
}

template <bool Upper, bool Unit,typename TriangularA, typename MatB>
void trsm(
	matrix_expression<TriangularA> const& A,
	matrix_expression<MatB>& B,
	boost::mpl::false_
){
	trsm_impl<Unit>(
		A,B,
		boost::mpl::bool_<Upper>(),
		typename TriangularA::orientation()
	);
}

}}}
#endif
