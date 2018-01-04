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
#ifndef REMORA_KERNELS_CLBLAST_TRMM_HPP
#define REMORA_KERNELS_CLBLAST_TRMM_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <clblast.h>
namespace remora{ namespace kernels{

// C <- AC with A being triangular
template <bool Upper,bool Unit,typename MatA, typename MatC>
void trmm(
	matrix_expression<MatA, gpu_tag> const& A, 
	matrix_expression<MatC, gpu_tag>& C
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size2() == C().size1());
	
	static_assert(std::is_same<typename MatA::value_type, typename MatC::value_type>::value, "[trmm] Arguments do not have same element type");
	static_assert(std::is_same<typename MatA::evaluation_category::tag, dense_tag>::value, "[trmm] A is not dense");
	static_assert(std::is_base_of<dense_tag, typename MatC::storage_type::storage_tag>::value, "[trmm] C does not have dense storage layout");
	
	//pre-evaluate A into a temporary if necessary
	auto const& Aeval = eval_expression(A);
	
	using namespace clblast;
	
	//obtain geometry information
	auto transA = std::is_same<typename MatA::orientation,typename MatC::orientation>::value? Transpose::kNo : Transpose::kYes;
	auto layout = std::is_same<typename MatC::orientation::orientation, row_major>::value? Layout::kRowMajor : Layout::kColMajor; 
	auto diagonal = Unit? Diagonal::kUnit : Diagonal::kNonUnit; 
	auto triangular = Upper? Triangle::kUpper : Triangle::kLower; 
	if(transA == Transpose::kYes){//when we transpose the matrix, we also have to change its Triangular type
		triangular = Upper? Triangle::kLower : Triangle::kUpper; 
	}
	std::size_t m = C().size1();
	std::size_t n = C().size2();
	
	//obtain raw storage
	auto storageA = Aeval.raw_storage();
	auto storageC = C().raw_storage();
	
	cl_event* event = nullptr;//todo: store events for out-of-order queues 
	auto code = Trmm(layout, Side::kLeft, triangular, transA, diagonal,
		m, n, typename MatC::value_type(1),
		storageA.buffer.get(), storageA.offset, storageA.leading_dimension,
		storageC.buffer.get(), storageC.offset, storageC.leading_dimension,
                &C().queue().get(), event
	);
	assert(code == StatusCode::kSuccess);
}

}}

#endif
