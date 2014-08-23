/*!
 * 
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

#ifndef SHARK_LINALG_BLAS_KERNELS_GOTOBLAS_TRSV_H
#define SHARK_LINALG_BLAS_KERNELS_GOTOBLAS_TRSV_H

#include "cblas_inc.h"

///solves systems of triangular matrices

namespace shark {namespace detail {namespace bindings {
inline void trsv(
	CBLAS_ORDER order, CBLAS_UPLO uplo,
	CBLAS_TRANSPOSE transA, CBLAS_DIAG unit,
	int n,
	float const *matA, int lda, float *vecB, int strideB
){
	cblas_strsv(order, uplo, transA, unit,n, const_cast<float*>(matA), lda, vecB, strideB);
}

inline void trsv(
	CBLAS_ORDER order, CBLAS_UPLO uplo,
	CBLAS_TRANSPOSE transA, CBLAS_DIAG unit,
	int n,
	double const *matA, int lda, double *vecB, int strideB
){
	cblas_dtrsv(order, uplo, transA, unit,n, const_cast<double*>(matA), lda, vecB, strideB);
}

inline void trsv(
	CBLAS_ORDER order, CBLAS_UPLO uplo,
	CBLAS_TRANSPOSE transA, CBLAS_DIAG unit,
	int n,
	std::complex<float> const *matA, int lda, std::complex<float> *vecB, int strideB
){
	cblas_ctrsv(order, uplo, transA, unit,n,
	        (float*)(matA), lda,
	        (float*)(vecB), strideB);
}
inline void trsv(
	CBLAS_ORDER order, CBLAS_UPLO uplo,
	CBLAS_TRANSPOSE transA, CBLAS_DIAG unit,
	int n,
	std::complex<double> const *matA, int lda, std::complex<double> *vecB, int strideB
){
	cblas_ztrsv(order, uplo, transA, unit,n,
	        (double*)(matA), lda,
	        (double*)(vecB), strideB);
}
// trsv(): solves matA system of linear equations matA * x = b
//             when matA is matA triangular matrix.
template <typename SymmA, typename VecB>
void trsv(
	CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA, CBLAS_DIAG unit,
	blas::matrix_expression<SymmA> const &matA, 
	blas::vector_expression<VecB> &vecB
){
	CBLAS_ORDER const storOrd= (CBLAS_ORDER)storage_order<typename SymmA::orientation>::value;

	SIZE_CHECK(matA().size1() == matA().size2());
	SIZE_CHECK(matA().size1()== vecB().size());

	int const n = matA().size1();

	trsv(storOrd, uplo, transA,unit, n,
	        traits::matrix_storage(matA()),
	        traits::leadingDimension(matA()),
	        traits::vector_storage(vecB()),
	        traits::vector_stride(vecB()));
}

template <bool upper,bool unit,typename SymmA, typename VecB>
void trsv(
	blas::matrix_expression<SymmA> const &matA, 
	blas::vector_expression<VecB> &vecB
){
	CBLAS_DIAG cblasUnit = unit?CblasUnit:CblasNonUnit;
	if(traits::isTransposed(matA))
		trsv(upper?CblasLower:CblasUpper,CblasTrans,cblasUnit, trans(matA),vecB);
	else
		trsv(upper?CblasUpper:CblasLower,CblasNoTrans,cblasUnit, matA,vecB);
}

}}}
#endif
