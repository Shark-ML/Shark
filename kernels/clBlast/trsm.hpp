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
#ifndef REMORA_KERNELS_CLBLAST_TRSM_HPP
#define REMORA_KERNELS_CLBLAST_TRSM_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include "../../proxy_expressions.hpp"
#include <clblast.h>
namespace remora{ namespace kernels{
	
	
template <class Triangular, typename MatA, typename MatB>
void trsm_impl(
	matrix_expression<MatA, gpu_tag> const& A, 
	matrix_expression<MatB, gpu_tag>& B,
	Triangular,
	left
){	
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size2() == B().size1());
	
	static_assert(std::is_same<typename MatA::value_type, typename MatB::value_type>::value, "[trsm] Arguments do not have same element type");
	static_assert(std::is_same<typename MatA::evaluation_category::tag, dense_tag>::value, "[trsm] A is not dense");
	static_assert(std::is_base_of<dense_tag, typename MatB::storage_type::storage_tag>::value, "[trsm] B does not have dense storage layout");
	
	//pre-evaluate A into a temporary if necessary
	auto const& Aeval = eval_expression(A);
	
	using namespace clblast;
	
	//obtain geometry information
	auto transA = std::is_same<typename MatA::orientation,typename MatB::orientation>::value? Transpose::kNo : Transpose::kYes;
	auto layout = std::is_same<typename MatB::orientation::orientation, row_major>::value? Layout::kRowMajor : Layout::kColMajor; 
	auto diagonal = Triangular::is_unit? Diagonal::kUnit : Diagonal::kNonUnit; 
	auto triangular = Triangular::is_upper? Triangle::kUpper : Triangle::kLower; 
	if(transA == Transpose::kYes){//when we transpose the matrix, we also have to change its Triangular type
		triangular = Triangular::is_upper? Triangle::kLower : Triangle::kUpper; 
	}
	std::size_t m = B().size1();
	std::size_t n = B().size2();
	
	//obtain raw storage
	auto storageA = Aeval.raw_storage();
	auto storageB = B().raw_storage();
	
	cl_event* event = nullptr;//todo: store events for out-of-order queues 
	auto code = Trsm(layout, Side::kLeft, triangular, transA, diagonal,
		m, n, typename MatB::value_type(1),
		storageA.buffer.get(), storageA.offset, storageA.leading_dimension,
		storageB.buffer.get(), storageB.offset, storageB.leading_dimension,
               &B().queue().get(), event
	);
	assert(code == StatusCode::kSuccess);
}


template <class Triangular, typename MatA, typename MatB>
void trsm_impl(
	matrix_expression<MatA, gpu_tag> const& A, 
	matrix_expression<MatB, gpu_tag>& B,
	Triangular,
	right
){	
	auto transB = trans(B);
	trsm_impl(trans(A), transB, typename Triangular::transposed_orientation(),  left());
}

// solve AX = B or XA=B with A being triangular
template <class Triangular, class Side, typename MatA, typename MatB>
void trsm(
	matrix_expression<MatA, gpu_tag> const& A, 
	matrix_expression<MatB, gpu_tag>& B
){
	trsm_impl(A,B,Triangular(),Side());
}

}}

#endif
