//===========================================================================
/*!
 * 
 *
 * \brief       -
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
#ifndef REMORA_KERNELS_CLBLAST_GEMV_HPP
#define REMORA_KERNELS_CLBLAST_GEMV_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <clblast.h>
namespace remora{ namespace kernels{

// v <- v + alpha * A * x
template <typename MatA, typename VecX, typename VecV>
void gemv(
	matrix_expression<MatA, gpu_tag> const& A,
	vector_expression<VecX, gpu_tag> const& x,
	vector_expression<VecV, gpu_tag>& v, 
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
	
	
	using namespace clblast;
	
	//obtain geometry information
	auto layout = std::is_same<typename MatA::orientation::orientation, row_major>::value? Layout::kRowMajor: Layout::kColMajor; 
	std::size_t m = A().size1();
	std::size_t n = A().size2();
	
	//obtain raw storage
	auto storageA = Aeval.raw_storage();
	auto storagex = xeval.raw_storage();
	auto storagev = v().raw_storage();
	
	cl_event* event = nullptr;//todo: store events for out-of-order queues 
	auto code =  Gemv(layout, Transpose::kNo,
		m, n, alpha,
		storageA.buffer.get(), storageA.offset, storageA.leading_dimension,
		storagex.buffer.get(), storagex.offset, storagex.stride,
		typename VecV::value_type(1),
		storagev.buffer.get(), storagev.offset, storagev.stride,
		&v().queue().get(), event
	);
	assert(code == StatusCode::kSuccess);
}

}}

#endif
