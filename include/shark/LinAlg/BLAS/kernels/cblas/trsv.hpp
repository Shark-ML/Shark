/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2011
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#include <boost/mpl/bool.hpp>

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
template <class Triangular,typename MatA, typename V>
void trsv_impl(
	matrix_expression<MatA, cpu_tag> const &A, 
	vector_expression<V, cpu_tag> &b,
	boost::mpl::true_, left
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size1()== b().size());
	CBLAS_DIAG cblasUnit = Triangular::is_unit?CblasUnit:CblasNonUnit;
	CBLAS_ORDER const storOrd= (CBLAS_ORDER)storage_order<typename MatA::orientation>::value;
	CBLAS_UPLO uplo = Triangular::is_upper?CblasUpper:CblasLower;
	

	int const n = A().size1();
	auto storageA = A().raw_storage();
	auto storageb = b().raw_storage();
	trsv(storOrd, uplo, CblasNoTrans,cblasUnit, n,
	        storageA.values,
	        storageA.leading_dimension,
		storageb.values,
	        storageb.stride
	);
}

//right is mapped onto left via transposing A
template <class Triangular,typename MatA, typename V>
void trsv_impl(
	matrix_expression<MatA, cpu_tag> const &A, 
	vector_expression<V, cpu_tag> &b,
	boost::mpl::true_, right
){
	matrix_transpose<typename const_expression<MatA>::type> transA(A());
	trsv_impl<typename Triangular::transposed_orientation>(transA, b, boost::mpl::true_(),  left());
}

//dispatcher

template <class Triangular, class Side,typename MatA, typename V>
void trsv(
	matrix_expression<MatA, cpu_tag> const& A, 
	vector_expression<V, cpu_tag> & b,
	boost::mpl::true_//optimized
){
	trsv_impl<Triangular>(A,b,boost::mpl::true_(), Side());
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
	typename M::storage_type::storage_tag,
	typename V::storage_type::storage_tag,
	typename M::value_type,
	typename V::value_type
>{};

}}}
#endif
