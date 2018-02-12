//===========================================================================
/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2010
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
//===========================================================================
#ifndef REMORA_KERNELS_CBLAS_TRMM_HPP
#define REMORA_KERNELS_CBLAS_TRMM_HPP

#include "cblas_inc.hpp"
#include <type_traits>

namespace remora{namespace bindings {

inline void trmm(
	CBLAS_ORDER const order,
	CBLAS_SIDE const side,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const M,
	int const N,
	float const *A, int const lda,
        float* B, int const incB
) {
	cblas_strmm(order, side, uplo, transA, unit, M, N, 
		1.0,
		A, lda,
	        B, incB
	);
}

inline void trmm(
	CBLAS_ORDER const order,
	CBLAS_SIDE const side,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const M,
	int const N,
	double const *A, int const lda,
        double* B, int const incB
) {
	cblas_dtrmm(order, side, uplo, transA, unit, M, N, 
		1.0,
		A, lda,
	        B, incB
	);
}


inline void trmm(
	CBLAS_ORDER const order,
	CBLAS_SIDE const side,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const M,
	int const N,
	std::complex<float> const *A, int const lda,
        std::complex<float>* B, int const incB
) {
	std::complex<float> alpha = 1.0;
	cblas_ctrmm(order, side, uplo, transA, unit, M, N, 
		reinterpret_cast<cblas_float_complex_type const *>(&alpha),
		reinterpret_cast<cblas_float_complex_type const *>(A), lda,
	        reinterpret_cast<cblas_float_complex_type *>(B), incB
	);
}

inline void trmm(
	CBLAS_ORDER const order,
	CBLAS_SIDE const side,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const M,
	int const N,
	std::complex<double> const *A, int const lda,
        std::complex<double>* B, int const incB
) {
	std::complex<double> alpha = 1.0;
	cblas_ztrmm(order, side, uplo, transA, unit, M, N, 
		reinterpret_cast<cblas_double_complex_type const *>(&alpha),
		reinterpret_cast<cblas_double_complex_type const *>(A), lda,
	        reinterpret_cast<cblas_double_complex_type *>(B), incB
	);
}

template <bool upper, bool unit, typename MatA, typename MatB>
void trmm(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag>& B,
	std::true_type
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size2() == B().size1());
	std::size_t n = A().size1();
	std::size_t m = B().size2();
	CBLAS_DIAG cblasUnit = unit?CblasUnit:CblasNonUnit;
	CBLAS_UPLO cblasUplo = upper?CblasUpper:CblasLower;
	CBLAS_ORDER stor_ord= (CBLAS_ORDER)storage_order<typename MatA::orientation>::value;
	CBLAS_TRANSPOSE trans=CblasNoTrans;
	
	//special case: MatA and MatB do not have same storage order. in this case compute as
	//AB->B^TA^T where transpose of B is done implicitely by exchanging storage order
	CBLAS_ORDER stor_ordB= (CBLAS_ORDER)storage_order<typename MatB::orientation>::value;
	if(stor_ord != stor_ordB){
		trans = CblasTrans;
		cblasUplo=  upper?CblasLower:CblasUpper;
	}
	
	auto storageA = A().raw_storage();
	auto storageB = B().raw_storage();
	trmm(stor_ordB, CblasLeft, cblasUplo, trans, cblasUnit,
		(int)n, int(m),
		storageA.values,
	        storageA.leading_dimension,
		storageB.values,
	        storageB.leading_dimension
	);
}


template<class M1, class M2>
struct has_optimized_trmm: std::integral_constant<bool,
	allowed_cblas_type<typename M1::value_type>::type::value
	&& std::is_same<typename M1::value_type, typename M2::value_type>::value
	&& std::is_base_of<dense_tag, typename M1::storage_type::storage_tag>::value
	&& std::is_base_of<dense_tag, typename M2::storage_type::storage_tag>::value 
>{};

}}
#endif
