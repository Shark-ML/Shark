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
#ifndef REMORA_KERNELS_CBLAS_GEMV_HPP
#define REMORA_KERNELS_CBLAS_GEMV_HPP

#include "cblas_inc.hpp"
#include <type_traits>

namespace remora{namespace bindings {

inline void gemv(CBLAS_ORDER const Order,
        CBLAS_TRANSPOSE const TransA, int const M, int const N,
        double alpha, float const *A, int const lda,
        float const *X, int const incX,
        double beta, float *Y, int const incY
) {
	cblas_sgemv(Order, TransA, M, N, alpha, A, lda,
	        X, incX,
	        beta, Y, incY);
}

inline void gemv(CBLAS_ORDER const Order,
        CBLAS_TRANSPOSE const TransA, int const M, int const N,
        double alpha, double const *A, int const lda,
        double const *X, int const incX,
        double beta, double *Y, int const incY
) {
	cblas_dgemv(Order, TransA, M, N, alpha, A, lda,
	        X, incX,
	        beta, Y, incY);
}

inline void gemv(CBLAS_ORDER const Order,
        CBLAS_TRANSPOSE const TransA, int const M, int const N,
        double alpha,
        std::complex<float> const *A, int const lda,
        std::complex<float> const *X, int const incX,
        double beta,
        std::complex<float> *Y, int const incY
) {
	std::complex<float> alphaArg(alpha,0);
	std::complex<float> betaArg(beta,0);
	cblas_cgemv(Order, TransA, M, N,
	        reinterpret_cast<cblas_float_complex_type const *>(&alphaArg),
	        reinterpret_cast<cblas_float_complex_type const *>(A), lda,
	        reinterpret_cast<cblas_float_complex_type const *>(X), incX,
	        reinterpret_cast<cblas_float_complex_type const *>(&betaArg),
	        reinterpret_cast<cblas_float_complex_type *>(Y), incY);
}

inline void gemv(CBLAS_ORDER const Order,
        CBLAS_TRANSPOSE const TransA, int const M, int const N,
         double alpha,
        std::complex<double> const *A, int const lda,
        std::complex<double> const *X, int const incX,
        double beta,
        std::complex<double> *Y, int const incY
) {
	std::complex<double> alphaArg(alpha,0);
	std::complex<double> betaArg(beta,0);
	cblas_zgemv(Order, TransA, M, N,
	        reinterpret_cast<cblas_double_complex_type const *>(&alphaArg),
	        reinterpret_cast<cblas_double_complex_type const *>(A), lda,
	        reinterpret_cast<cblas_double_complex_type const *>(X), incX,
	        reinterpret_cast<cblas_double_complex_type const *>(&betaArg),
	        reinterpret_cast<cblas_double_complex_type *>(Y), incY);
}


// y <- alpha * op (A) * x + beta * y
// op (A) == A || A^T || A^H
template <typename MatA, typename VectorX, typename VectorY>
void gemv(
	matrix_expression<MatA, cpu_tag> const &A,
	vector_expression<VectorX, cpu_tag> const &x,
        vector_expression<VectorY, cpu_tag> &y,
	typename VectorY::value_type alpha,
	std::true_type
){
	std::size_t m = A().size1();
	std::size_t n = A().size2();
	
	REMORA_SIZE_CHECK(x().size() == A().size2());
	REMORA_SIZE_CHECK(y().size() == A().size1());

	CBLAS_ORDER const stor_ord= (CBLAS_ORDER)storage_order<typename MatA::orientation>::value;
	
	auto storageA = A().raw_storage();
	auto storagex = x().raw_storage();
	auto storagey = y().raw_storage();
	gemv(stor_ord, CblasNoTrans, (int)m, (int)n, alpha,
		storageA.values,
	        storageA.leading_dimension,
		storagex.values,
	        storagex.stride,
	        typename VectorY::value_type(1),
		storagey.values,
	        storagey.stride
	);
}

template<class M, class V1, class V2>
struct has_optimized_gemv: std::integral_constant<bool,
	allowed_cblas_type<typename M::value_type>::type::value
	&& std::is_same<typename M::value_type, typename V1::value_type>::value
	&& std::is_same<typename V1::value_type, typename V2::value_type>::value
	&& std::is_base_of<dense_tag, typename M::storage_type::storage_tag>::value
	&& std::is_base_of<dense_tag, typename V1::storage_type::storage_tag>::value 
	&& std::is_base_of<dense_tag, typename V2::storage_type::storage_tag>::value 
>{};

}}
#endif
