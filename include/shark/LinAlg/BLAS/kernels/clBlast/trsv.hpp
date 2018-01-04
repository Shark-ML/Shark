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
#ifndef REMORA_KERNELS_CLBLAST_TRSV_HPP
#define REMORA_KERNELS_CLBLAST_TRSV_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <clblast.h>
namespace remora{ namespace kernels{

// solve Ax = b or xA=b with A being triangular
template <class Triangular,class Side, typename MatA, typename VecB>
void trsv(
	matrix_expression<MatA, gpu_tag> const& A, 
	vector_expression<VecB, gpu_tag>& b
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size1() == b().size());
	
	static_assert(std::is_same<typename MatA::value_type, typename VecB::value_type>::value, "[trsv] Arguments do not have same element type");
	static_assert(std::is_same<typename MatA::evaluation_category::tag, dense_tag>::value, "[trsv] A is not dense");
	static_assert(std::is_base_of<dense_tag, typename VecB::storage_type::storage_tag>::value, "[trsv] b does not have dense storage layout");
	
	//pre-evaluate A into a temporary if necessary
	auto const& Aeval = eval_expression(A);
	
	using namespace clblast;
	
	//obtain geometry information
	auto layout = std::is_same<typename MatA::orientation::orientation, row_major>::value? Layout::kRowMajor : Layout::kColMajor; 
	auto triangular = Triangular::is_upper? Triangle::kUpper : Triangle::kLower;
	auto diagonal = Triangular::is_unit? Diagonal::kUnit : Diagonal::kNonUnit; 
	//transpose if side is right
	if(!Side::is_left){
		layout = (layout == Layout::kRowMajor) ?  Layout::kColMajor :  Layout::kRowMajor;
		triangular = Triangular::is_upper? Triangle::kLower : Triangle::kUpper; 
	}
	std::size_t n = A().size1();
	
	//obtain raw storage
	auto storageA = Aeval.raw_storage();
	auto storageb = b().raw_storage();
	
	cl_event* event = nullptr;//todo: store events for out-of-order queues 
	auto code =  Trsv<typename VecB::value_type>(layout, triangular, Transpose::kNo , diagonal,
                n,
		storageA.buffer.get(), storageA.offset, storageA.leading_dimension,
		storageb.buffer.get(), storageb.offset, storageb.stride,
                &b().queue().get(), event
	);
	assert(code == StatusCode::kSuccess);
}

}}

#endif
