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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_LINALG_BLAS_KERNELS_GOTOBLAS_GEMV_H
#define SHARK_LINALG_BLAS_KERNELS_GOTOBLAS_GEMV_H

#include "cblas_inc.h"

namespace shark {namespace detail {namespace bindings {

inline void gemv(CBLAS_ORDER const Order,
        CBLAS_TRANSPOSE const TransA, int const M, int const N,
        double alpha, float const *A, int const lda,
        float const *X, int const incX,
        double beta, float *Y, int const incY
) {
	cblas_sgemv(Order, TransA, M, N, alpha, 
			const_cast<float*>(A), lda,
	        const_cast<float*>(X), incX,
	        beta, 
	        const_cast<float*>(Y), incY);
}

inline void gemv(CBLAS_ORDER const Order,
        CBLAS_TRANSPOSE const TransA, int const M, int const N,
        double alpha, double const *A, int const lda,
        double const *X, int const incX,
        double beta, double *Y, int const incY
) {
	cblas_dgemv(Order, TransA, M, N, alpha, 
			const_cast<double*>(A), lda,
	        const_cast<double*>(X), incX,
	        beta, const_cast<double*>(Y), incY);
}

inline void gemv(CBLAS_ORDER const Order,
        CBLAS_TRANSPOSE const TransA, int const M, int const N,
        double alpha,
        std::complex<float> const *A, int const lda,
        std::complex<float> const *X, int const incX,
        double beta,
        std::complex<float> *Y, int const incY
) {
	std::complex<float> alphaArg(alpha,0);
	std::complex<float> betaArg(beta,0);
	cblas_cgemv(Order, TransA, M, N,
	        (float*)(&alphaArg),
	        (float*)(A), lda,
	        (float*)(X), incX,
	        (float*)(&betaArg),
	        (float*)(Y), incY);
}

inline void gemv(CBLAS_ORDER const Order,
        CBLAS_TRANSPOSE const TransA, int const M, int const N,
         double alpha,
        std::complex<double> const *A, int const lda,
        std::complex<double> const *X, int const incX,
        double beta,
        std::complex<double> *Y, int const incY
) {
	std::complex<double> alphaArg(alpha,0);
	std::complex<double> betaArg(beta,0);
	cblas_zgemv(Order, TransA, M, N,
	        (double*)(&alphaArg),
	        (double*)(A), lda,
	        (double*)(X), incX,
	        (double*)(&betaArg),
	        (double*)(Y), incY);
}

// y <- alpha * op (A) * x + beta * y
// op (A) == A || A^T || A^H
template <typename Matr, typename VctX, typename VctY>
inline void gemv(CBLAS_TRANSPOSE const TransA,
        double alpha, blas::matrix_expression<Matr> const &a, blas::vector_expression<VctX> const &x,
        double beta, blas::vector_expression<VctY> &y
) {
	std::size_t m = a().size1();
	std::size_t n = a().size2();
	
	SIZE_CHECK(x().size() == (TransA == CblasNoTrans ? n : m));
	SIZE_CHECK(y().size() == (TransA == CblasNoTrans ? m : n));

	CBLAS_ORDER const stor_ord
	    = (CBLAS_ORDER)storage_order<typename traits::Orientation<Matr>::type>::value;
	
	gemv(stor_ord, TransA, (int)m, (int)n, alpha,
	        traits::matrix_storage(a()),
			traits::matrix_stride1(a()),
	        traits::vector_storage(x()),
	        traits::vector_stride(x()),
	        beta,
	        traits::vector_storage(y()),
	        traits::vector_stride(y()));
}

// y <- alpha * op (A) * x + beta * y
// op (A) == A || A^T || A^H
template <typename T, typename MatrA, typename VectorX, typename VectorY>
void gemv(T alpha, blas::matrix_expression<MatrA> const &matA, blas::vector_expression<VectorX> const &vecX,
        T beta, blas::vector_expression<VectorY> &vecY
){
	if(traits::isTransposed(matA))
		gemv(CblasTrans,alpha,trans(matA),vecX,beta,vecY);
	else
		gemv(CblasNoTrans,alpha,matA,vecX,beta,vecY);
}

}}}
#endif
