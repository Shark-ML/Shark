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
#ifndef REMORA_KERNELS_HIP_TRSM_HPP
#define REMORA_KERNELS_HIP_TRSM_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <clblast.h>

#ifdef __NVCC__
#include "../../hip/cublas_backend.hpp"
#endif

namespace remora{ namespace kernels{

// C <- AC with A being triangular
template <bool Upper,bool Unit, bool Left, typename MatA, typename MatB>
void trsm_impl(
	matrix_expression<MatA, hip_tag> const& A, 
	matrix_expression<MatB, hip_tag>& B,
	column_major
){
	
	//obtain geometry information
	auto transA = !std::is_same<typename MatA::orientation,typename MatB::orientation>::value;
	
	//obtain raw storage
	auto storageA = A().raw_storage();
	auto storageB = B().raw_storage();
	
	if(!transA){
		hip::get_blas(B().queue()).trsm(
			Left, Upper, transA, Unit,
			B().size1(), B().size2(),
			typename MatB::value_type(1),
			storageA.values, storageA.leading_dimension,
			storageB.values, storageB.leading_dimension,
			hip::get_stream(B().queue())
		);
	}else{
		hip::get_blas(B().queue()).trsm(
			!Left, Upper, transA, Unit,
			B().size2(), B().size1(),
			typename MatB::value_type(1),
			storageA.values, storageA.leading_dimension,
			storageB.values, storageB.leading_dimension,
			hip::get_stream(B().queue())
		);
	}
}

template <bool Upper,bool Unit, bool Left, typename MatA, typename MatB>
void trsm_impl(
	matrix_expression<MatA, hip_tag> const& A, 
	matrix_expression<MatB, hip_tag>& B,
	row_major
) {
	auto transB = trans(B);
	trsm_impl<!Upper, Unit, !Left>(trans(A), transB, column_major());
}

template <class Triangular, class Side, typename MatA, typename MatB>
void trsm(
	matrix_expression<MatA, hip_tag> const& A, 
	matrix_expression<MatB, hip_tag>& B
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(Side::is_left || A().size2() == B().size2());
	REMORA_SIZE_CHECK(!Side::is_left || A().size2() == B().size1());
	
	static_assert(std::is_same<typename MatA::value_type, typename MatB::value_type>::value, "[trsm] Arguments do not have same element type");
	static_assert(std::is_same<typename MatA::evaluation_category::tag, dense_tag>::value, "[trsm] A is not dense");
	static_assert(std::is_base_of<dense_tag, typename MatB::storage_type::storage_tag>::value, "[trsm] B does not have dense storage layout");
	
	//pre-evaluate A into a temporary if necessary
	auto const& Aeval = eval_expression(A);
	
	trsm_impl<Triangular::is_upper, Triangular::is_unit, Side::is_left>(Aeval, B, typename MatA::orientation());
}

}}

#endif
