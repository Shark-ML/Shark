//===========================================================================
/*!
 *  \author O. Krause
 *  \date 2011
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

#ifndef SHARK_LINALG_IMPL_NUMERIC_BINDINGS_ATLAS_POTRS_H
#define SHARK_LINALG_IMPL_NUMERIC_BINDINGS_ATLAS_POTRS_H
#include "cblas_inc.h"

namespace shark {namespace blas {namespace bindings {

inline int potrs(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
	int N, int NRHS,
	float const *A, int lda, float *B, int ldb
){
	return clapack_spotrs(Order, Uplo, N, NRHS, A, lda, B, ldb);
}

inline int potrs(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
	int N, int NRHS,
	double const *A, int lda, double *B, int ldb
){
	return clapack_dpotrs(Order, Uplo, N, NRHS, A, lda, B, ldb);
}

inline int potrs(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
        int N, int NRHS,
        std::complex<float> const *A, int lda,
        std::complex<float> *B, int ldb
){
	return clapack_cpotrs(Order, Uplo, N, NRHS,
		static_cast<void const *>(A), lda,
		static_cast<void *>(B), ldb
	);
}

inline int potrs(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
	int N, int NRHS,
	std::complex<double> const *A, int lda,
	std::complex<double> *B, int ldb
) {
	return clapack_zpotrs(Order, Uplo, N, NRHS,
		static_cast<void const *>(A), lda,
		static_cast<void *>(B), ldb
	);
}

// potrs(): solves a system of linear equations A * X = B
//          using the Cholesky factorization computed by potrf()
template <typename SymmA, typename MatrB>
int potrs(CBLAS_UPLO  uplo, matrix_expression<SymmA> const &a, matrix_expression<MatrB> &b) {
	CBLAS_ORDER stor_ord= (CBLAS_ORDER)storage_order<typename SymmA::orientation_category>::value;

	SIZE_CHECK(a().size1() == a().size2());

	int n = a().size1();
	int nrhs = b().size1();
	
	SIZE_CHECK(a().size1() == b().size1());

	return potrs(stor_ord, uplo, n, nrhs,
	        traits::matrix_storage(a()),
	        traits::leadingDimension(a()),
	        traits::matrix_storage(b()),
	        traits::leadingDimension(b()));
}

}}}
#endif
