//===========================================================================
/*!
 * 
 * \file        atlas/potrf.hpp
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2011
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
//===========================================================================

#ifndef SHARK_LINALG_IMPL_NUMERIC_BINDINGS_ATLAS_POTRF_H
#define SHARK_LINALG_IMPL_NUMERIC_BINDINGS_ATLAS_POTRF_H

#include "cblas_inc.hpp"

namespace shark {namespace blas {namespace bindings {

inline int potrf(CBLAS_ORDER const Order, CBLAS_UPLO const Uplo,
        int const N, float *A, int const lda
) {
	return clapack_spotrf(Order, Uplo, N, A, lda);
}

inline int potrf(CBLAS_ORDER const Order, CBLAS_UPLO const Uplo,
        int const N, double *A, int const lda
) {
	return clapack_dpotrf(Order, Uplo, N, A, lda);
}

inline int potrf(CBLAS_ORDER const Order, CBLAS_UPLO const Uplo,
        int const N, std::complex<float>* A, int const lda
) {
	return clapack_cpotrf(Order, Uplo, N, static_cast<void *>(A), lda);
}

inline int potrf(CBLAS_ORDER const Order, CBLAS_UPLO const Uplo,
        int const N, std::complex<double>* A, int const lda
) {
	return clapack_zpotrf(Order, Uplo, N, static_cast<void *>(A), lda);
}

template <typename SymmA>
inline int potrf(CBLAS_UPLO const uplo, matrix_container<SymmA>& a) {
	CBLAS_ORDER const stor_ord= 
		(CBLAS_ORDER)storage_order<typename SymmA::orientation>::value;

	std::size_t n = a().size1();
	SIZE_CHECK(n == a().size2());

	return potrf(stor_ord, uplo, (int)n,
	        traits::storage(a()),
	        traits::leading_dimension(a()));
}


}}}
#endif
