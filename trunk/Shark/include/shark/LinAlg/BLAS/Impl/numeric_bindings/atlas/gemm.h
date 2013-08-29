//===========================================================================
/*!
 *  \author O. Krause
 *  \date 2010
 *
 *  \par Copyright (c) 1998-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
//  Based on the boost::numeric bindings
/*
 *
 * Copyright (c) Kresimir Fresl 2002
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * Author acknowledges the support of the Faculty of Civil Engineering,
 * University of Zagreb, Croatia.
 *
 */

#ifndef SHARK_LINALG_IMPL_NUMERIC_BINDINGS_ATLAS_GEMM_HPP
#define SHARK_LINALG_IMPL_NUMERIC_BINDINGS_ATLAS_GEMM_HPP

#include "cblas_inc.h"

namespace shark { namespace blas { namespace bindings {

inline void gemm(CBLAS_ORDER const Order,
	CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	int M, int N, int K,
	float alpha, float const *A, int lda,
	float const *B, int ldb,
	float beta, float *C, int ldc
){
	cblas_sgemm(
		Order, TransA, TransB, M, N, K,
		alpha, A, lda,
		B, ldb,
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
		A, lda,
		B, ldb,
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
		static_cast<void const *>(&alphaArg),
		static_cast<void const *>(A), lda,
		static_cast<void const *>(B), ldb,
		static_cast<void const *>(&betaArg),
		static_cast<void *>(C), ldc
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
		static_cast<void const *>(&alphaArg),
		static_cast<void const *>(A), lda,
		static_cast<void const *>(B), ldb,
		static_cast<void const *>(&betaArg),
		static_cast<void *>(C), ldc
	);
}

// C <- alpha * op (A) * op (B) + beta * C
// op (A) == A || A^T || A^H
template <typename T, typename MatrA, typename MatrB, typename MatrC>
void gemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	T const &alpha, matrix_expression<MatrA> const &a, matrix_expression<MatrB> const &b,
	T const &beta, matrix_expression<MatrC>& c
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
		= (CBLAS_ORDER) storage_order<typename MatrC::orientation_category >::value;

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
void gemm(T const &alpha, matrix_expression<MatrA> const &a, matrix_expression<MatrB> const &b,
	T const &beta, matrix_expression<MatrC>& c
) {
	CBLAS_TRANSPOSE transposeA = traits::isTransposed(a)?CblasTrans:CblasNoTrans;
	CBLAS_TRANSPOSE transposeB = traits::isTransposed(b)?CblasTrans:CblasNoTrans;
	gemm(transposeA,transposeB,alpha,a,b,beta,c);
}
}}}

#endif
