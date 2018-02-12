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
#ifndef REMORA_KERNELS_CBLAS_TPMV_HPP
#define REMORA_KERNELS_CBLAS_TPMV_HPP

#include "cblas_inc.hpp"
#include <type_traits>

namespace remora{namespace bindings {

inline void tpmv(
	CBLAS_ORDER const Order,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const N,
	float const *A,
        float* X, int const incX
) {
	cblas_stpmv(Order, uplo, transA, unit, N, 
		A,
	        X, incX
	);
}

inline void tpmv(
	CBLAS_ORDER const Order,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const N,
	double const *A,
        double* X, int const incX
) {
	cblas_dtpmv(Order, uplo, transA, unit, N, 
		A,
	        X, incX
	);
}


inline void tpmv(
	CBLAS_ORDER const Order,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const N,
	std::complex<float> const *A,
        std::complex<float>* X, int const incX
) {
	cblas_ctpmv(Order, uplo, transA, unit, N, 
		reinterpret_cast<cblas_float_complex_type const *>(A),
	        reinterpret_cast<cblas_float_complex_type *>(X), incX
	);
}

inline void tpmv(
	CBLAS_ORDER const Order,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const N,
	std::complex<double> const *A,
        std::complex<double>* X, int const incX
) {
	cblas_ztpmv(Order, uplo, transA, unit, N, 
		reinterpret_cast<cblas_double_complex_type const *>(A),
	        reinterpret_cast<cblas_double_complex_type *>(X), incX
	);
}

template <typename MatA, typename VectorX>
void tpmv(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<VectorX, cpu_tag> &x,
	std::true_type
){
	REMORA_SIZE_CHECK(x().size() == A().size2());
	REMORA_SIZE_CHECK(A().size2() == A().size1());
	bool upper = MatA::orientation::triangular_type::is_upper;
	bool unit = MatA::orientation::triangular_type::is_unit;
	std::size_t n = A().size1();
	CBLAS_DIAG cblasUnit = unit?CblasUnit:CblasNonUnit;
	CBLAS_UPLO cblasUplo = upper?CblasUpper:CblasLower;
	CBLAS_ORDER stor_ord= (CBLAS_ORDER)storage_order<typename MatA::orientation::orientation>::value;
	
	auto storageA = A().raw_storage();
	auto storagex = x().raw_storage();
	tpmv(stor_ord, cblasUplo, CblasNoTrans, cblasUnit, (int)n,
	        storageA.values,
		storagex.values,
	        storagex.stride
	);
}

template<class M, class V>
struct has_optimized_tpmv: std::integral_constant<bool,
	allowed_cblas_type<typename M::value_type>::type::value
	&& std::is_same<typename M::value_type, typename V::value_type>::value
	&& std::is_base_of<packed_tag, typename M::storage_type::storage_tag>::value
	&& std::is_base_of<dense_tag, typename V::storage_type::storage_tag>::value 
>{};

}}
#endif
