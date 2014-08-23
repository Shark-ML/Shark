/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2010
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

#ifndef SHARK_LINALG_BLAS_KERNELS_GOTOBLAS_GEMM_HPP
#define SHARK_LINALG_BLAS_KERNELS_GOTOBLAS_GEMM_HPP

#include "cblas_inc.h"


namespace shark { namespace detail { namespace bindings {

inline void gemm(CBLAS_ORDER const Order,
	CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	int M, int N, int K,
	float alpha, float const *A, int lda,
	float const *B, int ldb,
	float beta, float *C, int ldc
){
	cblas_sgemm(
		Order, TransA, TransB, M, N, K,
		alpha, 
		const_cast<float*>(A), lda,
		const_cast<float*>(B), ldb,
		beta, C, ldc
	);
}

inline void gemm(CBLAS_ORDER const Order,
	CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	int M, int N, int K,
	double alpha, double const *A, int lda,
	double const *B, int ldb,
	double beta, double *C, int ldc
){
	cblas_dgemm(
		Order, TransA, TransB, M, N, K,
		alpha, 
		const_cast<double*>(A), lda,
		const_cast<double*>(B), ldb,
		beta, 
		C, ldc
	);
}

inline void gemm(CBLAS_ORDER const Order,
	CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	int M, int N, int K,
	float alpha,
	std::complex<float> const *A, int lda,
	std::complex<float> const *B, int ldb,
	float beta,
	std::complex<float>* C, int ldc
) {
	std::complex<float> alphaArg(alpha,0);
	std::complex<float> betaArg(beta,0);
	cblas_cgemm(
		Order, TransA, TransB, M, N, K,
		(float*)(&alphaArg),
		(float*)A, lda,
		(float*)B, ldb,
		(float*)(&betaArg),
		(float*)C, ldc
	);
}

inline void gemm(CBLAS_ORDER const Order,
	CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	int M, int N, int K,
	double alpha,
	std::complex<double> const *A, int lda,
	std::complex<double> const *B, int ldb,
	double beta,
	std::complex<double>* C, int ldc
) {
	std::complex<double> alphaArg(alpha,0);
	std::complex<double> betaArg(beta,0);
	cblas_zgemm(
		Order, TransA, TransB, M, N, K,
		(double*)(&alphaArg),
		(double*)(A), lda,
		(double*)(B), ldb,
		(double*)(&betaArg),
		(double*)(C), ldc
	);
}

// C <- alpha * op (A) * op (B) + beta * C
// op (A) == A || A^T || A^H
template <typename T, typename MatrA, typename MatrB, typename MatrC>
void gemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	T const &alpha, blas::matrix_expression<MatrA> const &a, blas::matrix_expression<MatrB> const &b,
	T const &beta, blas::matrix_expression<MatrC>& c
) {
	std::size_t m,n,k;
	if ((TransA == CblasNoTrans) != traits::isTransposed(a)) {
		m = a().size1();
		k = a().size2();
	} else {
		m = a().size2();
		k = a().size1();
	}
	
	if ((TransB== CblasNoTrans)  != traits::isTransposed(b)) {
		n = b().size2();
		SIZE_CHECK(k == b().size1());
	} else {
		n = b().size1();
		SIZE_CHECK(k == b().size2());
	}
	SIZE_CHECK(m ==  c().size1());
	SIZE_CHECK(n ==  c().size2());
	CBLAS_ORDER stor_ord
		= (CBLAS_ORDER) storage_order<typename MatrC::orientation >::value;

	gemm(stor_ord, TransA, TransB, (int)m, (int)n, (int)k, alpha,
		traits::matrix_storage(a()),
		traits::leadingDimension(a()),
		traits::matrix_storage(b()),
		traits::leadingDimension(b()),
		beta,
		traits::matrix_storage(c()),
		traits::leadingDimension(c()));
}
// C <- alpha * op (A) * op (B) + beta * C
// op (A) == A || A^T || A^H
template <typename T, typename MatrA, typename MatrB, typename MatrC>
void gemm(T const &alpha, blas::matrix_expression<MatrA> const &a, blas::matrix_expression<MatrB> const &b,
	T const &beta, blas::matrix_expression<MatrC>& c
) {
	CBLAS_TRANSPOSE transposeA = traits::isTransposed(a)?CblasTrans:CblasNoTrans;
	CBLAS_TRANSPOSE transposeB = traits::isTransposed(b)?CblasTrans:CblasNoTrans;
	gemm(transposeA,transposeB,alpha,a,b,beta,c);
}

}}}

#endif
