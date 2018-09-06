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
#ifndef REMORA_KERNELS_HIP_TRMV_HPP
#define REMORA_KERNELS_HIP_TRMV_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <clblast.h>

#ifdef __NVCC__
#include "../../hip/cublas_backend.hpp"
#endif
namespace remora{ namespace kernels{

// v <- Av with A being triangular
template <bool Upper,bool Unit,typename MatA, typename VecV>
void trmv(
	matrix_expression<MatA, hip_tag> const& A, 
	vector_expression<VecV, hip_tag>& v
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size2() == v().size());
	
	static_assert(std::is_same<typename MatA::value_type, typename VecV::value_type>::value, "[trmv] Arguments do not have same element type");
	static_assert(std::is_same<typename MatA::evaluation_category::tag, dense_tag>::value, "[trmv] A is not dense");
	static_assert(std::is_base_of<dense_tag, typename VecV::storage_type::storage_tag>::value, "[trmv] v does not have dense storage layout");
	
	//pre-evaluate A into a temporary if necessary
	auto const& Aeval = eval_expression(A);

	//obtain geometry information
	auto transA = std::is_same<typename MatA::orientation::orientation, row_major>::value;
	bool triangular = transA? !Upper : Upper;
	std::size_t n = A().size1();
	
	//obtain raw storage
	auto storageA = Aeval.raw_storage();
	auto storagev = v().raw_storage();
	
	hip::get_blas(v().queue()).trmv(
		triangular, transA, Unit,
		n,
		storageA.values, storageA.leading_dimension,
		storagev.values, storagev.stride,
		hip::get_stream(v().queue())
	);
}
}}

#endif
