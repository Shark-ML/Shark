/*!
 * 
 * \file        atlas/trsm.hpp
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

#ifndef SHARK_LINALG_BLAS_UBLAS_KERNELS_ATLAS_TRSM_HPP
#define SHARK_LINALG_BLAS_UBLAS_KERNELS_ATLAS_TRSM_HPP

#include "cblas_inc.hpp"

///solves systems of triangular matrices

namespace shark {namespace blas {namespace bindings {
inline void trsm(
	CBLAS_ORDER order, CBLAS_UPLO uplo,CBLAS_TRANSPOSE transA, 
	CBLAS_SIDE side, CBLAS_DIAG unit,
	int n, int nRHS,
	float const *A, int lda, float *B, int ldb
) {
	cblas_strsm(order, side, uplo, transA, unit,n, nRHS, 1.0, A, lda, B, ldb);
}

inline void trsm(
	CBLAS_ORDER order, CBLAS_UPLO uplo,CBLAS_TRANSPOSE transA, 
	CBLAS_SIDE side, CBLAS_DIAG unit,
	int n, int nRHS,
	double const *A, int lda, double *B, int ldb
) {
	cblas_dtrsm(order, side, uplo, transA, unit,n, nRHS, 1.0, A, lda, B, ldb);
}

inline void trsm(
	CBLAS_ORDER order, CBLAS_UPLO uplo,CBLAS_TRANSPOSE transA, 
	CBLAS_SIDE side, CBLAS_DIAG unit,
	int n, int nRHS,
	std::complex<float> const *A, int lda, std::complex<float> *B, int ldb
) {
	std::complex<float> alpha(1.0,0);
	cblas_ctrsm(order, side, uplo, transA, unit,n, nRHS,
		static_cast<void const *>(&alpha),
	        static_cast<void const *>(A), lda,
	        static_cast<void *>(B), ldb);
}
inline void trsm(
	CBLAS_ORDER order, CBLAS_UPLO uplo,CBLAS_TRANSPOSE transA, 
	CBLAS_SIDE side, CBLAS_DIAG unit,
	int n, int nRHS,
	std::complex<double> const *A, int lda, std::complex<double> *B, int ldb
) {
	std::complex<double> alpha(1.0,0);
	cblas_ztrsm(order, side, uplo, transA, unit,n, nRHS,
		static_cast<void const *>(&alpha),
	        static_cast<void const *>(A), lda,
	        static_cast<void *>(B), ldb);
}

// trsm(): solves A system of linear equations A * X = B
//             when A is A triangular matrix
template <bool upper, bool unit,typename TriangularA, typename MatB>
void trsm(
	matrix_expression<TriangularA> const &A,
	matrix_expression<MatB> &B,
	boost::mpl::true_
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size1() == B().size1());
	
	//orientation is defined by the second argument
	CBLAS_ORDER const storOrd = (CBLAS_ORDER)storage_order<typename MatB::orientation>::value;
	//if orientations do not match, wecan interpret this as transposing A
	bool transposeA =  !traits::same_orientation(A,B);
	
	CBLAS_DIAG cblasUnit = unit?CblasUnit:CblasNonUnit;
	CBLAS_UPLO cblasUplo = (upper != transposeA)?CblasUpper:CblasLower;
	
	
	CBLAS_TRANSPOSE transA = transposeA?CblasTrans:CblasNoTrans;
	
	int m = B().size1();
	int nrhs = B().size2();
	
	trsm(storOrd, cblasUplo, transA, CblasLeft,cblasUnit, m, nrhs,
		traits::storage(A),
		traits::leading_dimension(A),
		traits::storage(B),
		traits::leading_dimension(B)
	);
}

template<class Storage1, class Storage2, class T1, class T2>
struct optimized_trsm_detail{
	typedef boost::mpl::false_ type;
};
template<>
struct optimized_trsm_detail<
	dense_tag, dense_tag,
	double, double
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_trsm_detail<
	dense_tag, dense_tag, 
	float, float
>{
	typedef boost::mpl::true_ type;
};

template<>
struct optimized_trsm_detail<
	dense_tag, dense_tag,
	std::complex<double>, std::complex<double>
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_trsm_detail<
	dense_tag, dense_tag,
	std::complex<float>, std::complex<float>
>{
	typedef boost::mpl::true_ type;
};

template<class M1, class M2>
struct  has_optimized_trsm
: public optimized_trsm_detail<
	typename M1::storage_category,
	typename M2::storage_category,
	typename M1::value_type,
	typename M2::value_type
>{};

}}}
#endif
