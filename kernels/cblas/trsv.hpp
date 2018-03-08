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

#ifndef REMORA_KERNELS_CBLAS_TRSV_HPP
#define REMORA_KERNELS_CBLAS_TRSV_HPP

#include "cblas_inc.hpp"
#include <type_traits>

///solves systems of triangular matrices

namespace remora {namespace bindings {
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
	std::true_type, left
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size1()== b().size());
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
	std::true_type, right
){
	trsv_impl<typename Triangular::transposed_orientation>(trans(A), b, std::true_type(),  left());
}

//dispatcher

template <class Triangular, class Side,typename MatA, typename V>
void trsv(
	matrix_expression<MatA, cpu_tag> const& A, 
	vector_expression<V, cpu_tag> & b,
	std::true_type//optimized
){
	trsv_impl<Triangular>(A,b,std::true_type(), Side());
}

template<class M, class V>
struct has_optimized_trsv: std::integral_constant<bool,
	allowed_cblas_type<typename M::value_type>::type::value
	&& std::is_same<typename M::value_type, typename V::value_type>::value
	&& std::is_base_of<dense_tag, typename M::storage_type::storage_tag>::value
	&& std::is_base_of<dense_tag, typename V::storage_type::storage_tag>::value 
>{};

}}
#endif
