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

#ifndef SHARK_LINALG_BLAS_KERNELS_CBLAS_TRSV_HPP
#define SHARK_LINALG_BLAS_KERNELS_CBLAS_TRSV_HPP

#include "cblas_inc.hpp"

///solves systems of triangular matrices

namespace shark {namespace blas{ namespace bindings {
inline void trsv(
	CBLAS_ORDER order, CBLAS_UPLO uplo,
	CBLAS_TRANSPOSE transA, CBLAS_DIAG unit,
	int n,
	float const *A, int lda, float *b, int strideX
){
	cblas_strsv(order, uplo, transA, unit,n, A, lda, b, strideX);
}

inline void trsv(
	CBLAS_ORDER order, CBLAS_UPLO uplo,
	CBLAS_TRANSPOSE transA, CBLAS_DIAG unit,
	int n,
	double const *A, int lda, double *b, int strideX
){
	cblas_dtrsv(order, uplo, transA, unit,n, A, lda, b, strideX);
}

inline void trsv(
	CBLAS_ORDER order, CBLAS_UPLO uplo,
	CBLAS_TRANSPOSE transA, CBLAS_DIAG unit,
	int n,
	std::complex<float> const *A, int lda, std::complex<float> *b, int strideX
){
	cblas_ctrsv(order, uplo, transA, unit,n,
	        reinterpret_cast<cblas_float_complex_type const *>(A), lda,
	        reinterpret_cast<cblas_float_complex_type *>(b), strideX);
}
inline void trsv(
	CBLAS_ORDER order, CBLAS_UPLO uplo,
	CBLAS_TRANSPOSE transA, CBLAS_DIAG unit,
	int n,
	std::complex<double> const *A, int lda, std::complex<double> *b, int strideX
){
	cblas_ztrsv(order, uplo, transA, unit,n,
	        reinterpret_cast<cblas_double_complex_type const *>(A), lda,
	        reinterpret_cast<cblas_double_complex_type *>(b), strideX);
}

// trsv(): solves A system of linear equations A * x = b
//             when A is A triangular matrix.
template <bool Upper,bool Unit,typename TriangularA, typename V>
void trsv(
	matrix_expression<TriangularA> const &A, 
	vector_expression<V> &b,
	boost::mpl::true_
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size1()== b().size());
	CBLAS_DIAG cblasUnit = Unit?CblasUnit:CblasNonUnit;
	CBLAS_ORDER const storOrd= (CBLAS_ORDER)storage_order<typename TriangularA::orientation>::value;
	CBLAS_UPLO uplo = Upper?CblasUpper:CblasLower;
	

	int const n = A().size1();

	trsv(storOrd, uplo, CblasNoTrans,cblasUnit, n,
	        traits::storage(A),
	        traits::leading_dimension(A),
	        traits::storage(b),
	        traits::stride(b)
	);
}

template<class Storage1, class Storage2, class T1, class T2>
struct optimized_trsv_detail{
	typedef boost::mpl::false_ type;
};
template<>
struct optimized_trsv_detail<
	dense_tag, dense_tag,
	double, double
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_trsv_detail<
	dense_tag, dense_tag, 
	float, float
>{
	typedef boost::mpl::true_ type;
};

template<>
struct optimized_trsv_detail<
	dense_tag, dense_tag,
	std::complex<double>, std::complex<double>
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_trsv_detail<
	dense_tag, dense_tag,
	std::complex<float>, std::complex<float>
>{
	typedef boost::mpl::true_ type;
};

template<class M, class V>
struct  has_optimized_trsv
: public optimized_trsv_detail<
	typename M::storage_category,
	typename V::storage_category,
	typename M::value_type,
	typename V::value_type
>{};

}}}
#endif
