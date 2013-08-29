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

#ifndef SHARK_LINALG_IMPL_NUMERIC_BINDINGS_ATLAS_POTRF_H
#define SHARK_LINALG_IMPL_NUMERIC_BINDINGS_ATLAS_POTRF_H

#include "cblas_inc.h"

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
		(CBLAS_ORDER)storage_order<typename SymmA::orientation_category>::value;

	std::size_t n = a().size1();
	SIZE_CHECK(n == a().size2());

	return potrf(stor_ord, uplo, (int)n,
	        traits::matrix_storage(a()),
	        traits::leadingDimension(a()));
}


}}}
#endif
