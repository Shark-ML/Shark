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
#ifndef REMORA_KERNELS_HIP_GEMV_HPP
#define REMORA_KERNELS_HIP_GEMV_HPP

#include "../../proxy_expressions.hpp"
#include "../../hip/traits.hpp"

#ifdef __NVCC__
#include "../../hip/cublas_backend.hpp"
#endif

namespace remora{
namespace kernels{

// v <- v + alpha * A * x
template <typename MatA, typename VecX, typename VecV>
void gemv(
	matrix_expression<MatA, hip_tag> const& A,
	vector_expression<VecX, hip_tag> const& x,
	vector_expression<VecV, hip_tag>& v, 
	typename VecV::value_type const& alpha
) {
	REMORA_SIZE_CHECK(A().size1() == v().size());
	REMORA_SIZE_CHECK(A().size2() == x().size());
	
	static_assert(std::is_same<typename MatA::value_type, typename VecX::value_type>::value, "[gemv] Arguments do not have same element type");
	static_assert(std::is_same<typename MatA::value_type, typename VecV::value_type>::value, "[gemv] Arguments do not have same element type");	
	static_assert(std::is_same<typename MatA::evaluation_category::tag, dense_tag>::value, "[gemv] A is not dense");
	static_assert(std::is_same<typename VecX::evaluation_category::tag, dense_tag>::value, "[gemv] x is not dense");
	static_assert(std::is_base_of<dense_tag, typename VecV::storage_type::storage_tag>::value, "[gemv] v does not have dense storage layout");
	
	//pre-evaluate A and x into a temporary if necessary
	auto const& Aeval = eval_expression(A);
	auto const& xeval = eval_expression(x);
	
	//obtain geometry information
	bool transA = std::is_same<typename MatA::orientation, row_major>::value;
	std::size_t m = A().size1();
	std::size_t n = A().size2();
	if(transA)
		std::swap(m,n);

	//obtain matrix storage
	auto storageA = Aeval.raw_storage();
	auto storagex = xeval.raw_storage();
	auto storagev = v().raw_storage();
	
	hip::get_blas(A().queue()).gemv(
		transA,
		m, n,
		alpha,
		storageA.values, storageA.leading_dimension,
		storagex.values, storagex.stride,
		typename VecV::value_type(1),
		storagev.values, storagev.stride,
		hip::get_stream(A().queue())
	);
}

}}

#endif
