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
#ifndef REMORA_KERNELS_CBLAS_SYRK_HPP
#define REMORA_KERNELS_CBLAS_SYRK_HPP

#include "cblas_inc.hpp"
#include <type_traits>

namespace remora{ namespace bindings {

inline void syrk(
	CBLAS_ORDER const order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
	int N, int K,
	float alpha, float const *A, int lda,
	float beta, float *C, int ldc
){
	cblas_ssyrk(
		order, uplo, trans,
		N, K,
		alpha, A, lda,
		beta, C, ldc
	);
}

inline void syrk(
	CBLAS_ORDER const order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
	int N, int K,
	double alpha, double const *A, int lda,
	double beta, double *C, int ldc
){
	cblas_dsyrk(
		order, uplo, trans,
		N, K,
		alpha, A, lda,
		beta, C, ldc
	);
}


// C <- C + alpha * A * A^T 
template <bool Upper, typename MatA, typename MatC>
void syrk(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatC, cpu_tag>& C,
	typename MatC::value_type alpha,
	std::true_type
) {
	REMORA_SIZE_CHECK(A().size1() == C().size1());
	REMORA_SIZE_CHECK(C().size1() == C().size2());
	
	CBLAS_ORDER stor_ord = (CBLAS_ORDER) storage_order<typename MatC::orientation >::value;
	CBLAS_UPLO uplo = Upper?CblasUpper: CblasLower;
	CBLAS_TRANSPOSE trans = std::is_same<typename MatA::orientation,typename MatC::orientation>::value?CblasNoTrans:CblasTrans;
	std::size_t n = C().size1();
	std::size_t k = A().size2();
	

	auto storageA = A().raw_storage();
	auto storageC = C().raw_storage();
	syrk(stor_ord, uplo, trans,
		(int)n, (int)k, alpha,
		storageA.values,
	        storageA.leading_dimension,
		typename MatC::value_type(1),
		storageC.values,
	        storageC.leading_dimension
	);
}

template<class M1, class M2>
struct has_optimized_syrk: std::integral_constant<bool,
	allowed_cblas_type<typename M1::value_type>::type::value
	&& std::is_same<typename M1::value_type, typename M2::value_type>::value
	&& std::is_base_of<dense_tag, typename M1::storage_type::storage_tag>::value
	&& std::is_base_of<dense_tag, typename M2::storage_type::storage_tag>::value 
>{};

}}

#endif
