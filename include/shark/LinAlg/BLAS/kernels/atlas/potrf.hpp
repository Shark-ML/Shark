//===========================================================================
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
//===========================================================================

#ifndef SHARK_LINALG_BLAS_KERNELS_ATLAS_POTRF_H
#define SHARK_LINALG_BLAS_KERNELS_ATLAS_POTRF_H

#include "../cblas/cblas_inc.hpp"
extern "C"{
	#include <clapack.h>
}

namespace shark {
namespace blas {
namespace bindings {

inline int potrf(
	CBLAS_ORDER const Order, CBLAS_UPLO const Uplo,
        int const N, float *A, int const lda
) {
	return clapack_spotrf(Order, Uplo, N, A, lda);
}

inline int potrf(
	CBLAS_ORDER const Order, CBLAS_UPLO const Uplo,
        int const N, double *A, int const lda
) {
	return clapack_dpotrf(Order, Uplo, N, A, lda);
}

inline int potrf(
	CBLAS_ORDER const Order, CBLAS_UPLO const Uplo,
        int const N, std::complex<float>* A, int const lda
) {
	return clapack_cpotrf(Order, Uplo, N, static_cast<void *>(A), lda);
}

inline int potrf(
	CBLAS_ORDER const Order, CBLAS_UPLO const Uplo,
        int const N, std::complex<double>* A, int const lda
) {
	return clapack_zpotrf(Order, Uplo, N, static_cast<void *>(A), lda);
}

template <typename Triangular, typename SymmA>
inline int potrf(
	matrix_container<SymmA>& A,
	boost::mpl::true_
) {
	CBLAS_UPLO const uplo = Triangular::is_upper ? CblasUpper : CblasLower;
	CBLAS_ORDER const stor_ord =
		(CBLAS_ORDER)storage_order<typename SymmA::orientation>::value;

	std::size_t n = A().size1();
	SIZE_CHECK(n == A().size2());

	return potrf(
		stor_ord, uplo, (int)n,
	        traits::storage(A()),
	        traits::leading_dimension(A())
	);
}

template<class Storage, class T>
struct optimized_potrf_detail {
	typedef boost::mpl::false_ type;
};
template<>
struct optimized_potrf_detail <
	dense_tag,
	double
> {
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_potrf_detail <
	dense_tag,
	float
> {
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_potrf_detail <
	dense_tag,
	std::complex<double>
> {
	typedef boost::mpl::true_ type;
};

template<>
struct optimized_potrf_detail <
	dense_tag,
	std::complex<float>
> {
	typedef boost::mpl::true_ type;
};

template<class M>
struct  has_optimized_potrf
	: public optimized_potrf_detail <
	  typename M::storage_category,
	  typename M::value_type
	> {};
}}}
#endif
