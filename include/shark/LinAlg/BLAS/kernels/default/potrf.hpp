/*!
 *
 *
 * \brief       Implements the default implementation of the POTRF algorithm
 *
 * \author    O. Krause
 * \date        2014
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
#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_POTRF_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_POTRF_HPP

#include "../../matrix_proxy.hpp"
#include "../../vector_expression.hpp"
#include <boost/mpl/bool.hpp>

namespace shark {
namespace blas {
namespace bindings {


//upper potrf(row-major)
template<class MatA>
std::size_t potrf_impl(
    matrix_expression<MatA>& A,
    row_major, lower
) {
	std::size_t m = A().size1();
	for(size_t j = 0; j < m; j++) {
		for(size_t i = j; i < m; i++) {
			double s = A()(i, j);
			for(size_t k = 0; k < j; k++) {
				s -= A()(i, k) * A()(j, k);
			}
			if(i == j) {
				if(s <= 0)
					return i;
				A()(i, j) = std::sqrt(s);
			} else {
				A()(i, j) = s / A()(j , j);
			}
		}
	}
	return 0;
}

//lower potrf(row-major)
template<class MatA>
std::size_t potrf_impl(
    matrix_expression<MatA>& A,
    row_major, upper
) {
	std::size_t m = A().size1();
	for(size_t i = 0; i < m; i++) {
		double& Aii = A()(i, i);
		if(Aii < 0)
			return i;
		using std::sqrt;
		Aii = sqrt(Aii);
		//update row
		for(std::size_t j = i + 1; j < m; ++j) {
			A()(i, j) /= Aii;
		}
		//rank-one update
		for(size_t k = i + 1; k < m; k++) {
			for(std::size_t j = k; j < m; ++j) {
				A()(k, j) -= A()(i, k) * A()(i, j);
			}
		}
	}
	return 0;
}


//dispatcher for column major
template<class MatA, class Triangular>
std::size_t potrf_impl(
    matrix_container<MatA>& A,
    column_major, Triangular
) {
	blas::matrix_transpose<MatA> transA(A());
	return potrf_impl(transA, row_major(), typename Triangular::transposed_orientation());
}

//dispatcher

template <class Triangular, typename MatA>
std::size_t potrf(
    matrix_container<MatA>& A,
    boost::mpl::false_//unoptimized
) {
	return potrf_impl(A, typename MatA::orientation(), Triangular());
}

}
}
}
#endif
