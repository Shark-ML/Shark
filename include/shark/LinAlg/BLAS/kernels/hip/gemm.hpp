//===========================================================================
/*!
 * 
 *
 * \brief       Hip GEMM kernel frontend using cuBLAS or rocmBLAS backends
 *
 * \author      O. Krause
 * \date        2017
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
#ifndef REMORA_KERNELS_HIP_GEMM_HPP
#define REMORA_KERNELS_HIP_GEMM_HPP

#include "../../proxy_expressions.hpp"
#include "../../hip/traits.hpp"

#ifdef __NVCC__
#include "../../hip/cublas_backend.hpp"
#endif

namespace remora{
namespace kernels{

// C <- alpha * A * B + beta * C
template <typename MatA, typename MatB, typename MatC>
void gemm_impl(
	matrix_expression<MatA, hip_tag> const& A,
	matrix_expression<MatB, hip_tag> const& B,
	matrix_expression<MatC, hip_tag>& C, 
	typename MatC::value_type const& alpha,
	column_major
) {
	//obtain geometry information
	bool transA = !std::is_same<typename MatA::orientation,typename MatC::orientation>::value;
	bool transB = !std::is_same<typename MatB::orientation,typename MatC::orientation>::value;
	std::size_t m = C().size1();
	std::size_t n = C().size2();
	std::size_t k = A().size2();

	//obtain matrix storage
	auto storageA = A().raw_storage();
	auto storageB = B().raw_storage();
	auto storageC = C().raw_storage();
	
	hip::get_blas(C().queue()).gemm(
		transA, transB,
		m, n , k,
		alpha,
		storageA.values, storageA.leading_dimension,
		storageB.values, storageB.leading_dimension,
		typename MatC::value_type(1),
		storageC.values, storageC.leading_dimension,
		hip::get_stream(C().queue())
	);
}

template <typename MatA, typename MatB, typename MatC>
void gemm_impl(
	matrix_expression<MatA, hip_tag> const& A,
	matrix_expression<MatB, hip_tag> const& B,
	matrix_expression<MatC, hip_tag>& C, 
	typename MatC::value_type const& alpha,
	row_major
) {
	auto transC = trans(C);
	gemm_impl(trans(B), trans(A), transC, alpha, column_major());
}

template <typename MatA, typename MatB, typename MatC>
void gemm(
	matrix_expression<MatA, hip_tag> const& A,
	matrix_expression<MatB, hip_tag> const& B,
	matrix_expression<MatC, hip_tag>& C, 
	typename MatC::value_type const& alpha
) {
	REMORA_SIZE_CHECK(A().size1() == C().size1());
	REMORA_SIZE_CHECK(B().size2() == C().size2());
	REMORA_SIZE_CHECK(A().size2()== B().size1());
	
	static_assert(std::is_same<typename MatA::value_type, typename MatC::value_type>::value, "[gemm] Arguments do not have same element type");
	static_assert(std::is_same<typename MatA::value_type, typename MatB::value_type>::value, "[gemm] Arguments do not have same element type");
	static_assert(std::is_same<typename MatA::evaluation_category::tag, dense_tag>::value, "[gemm] A is not dense");
	static_assert(std::is_same<typename MatB::evaluation_category::tag, dense_tag>::value, "[gemm] B is not dense");
	static_assert(std::is_base_of<dense_tag, typename MatC::storage_type::storage_tag>::value, "[gemm] C does not have dense storage layout");
	
	//pre-evaluate A and B into a temporary if necessary
	auto const& Aeval = eval_expression(A);
	auto const& Beval = eval_expression(B);	
	
	gemm_impl(Aeval, Beval, C, alpha, typename MatC::orientation());
}

}}

#endif
