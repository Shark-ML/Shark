//===========================================================================
/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2016
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
#ifndef REMORA_KERNELS_HIP_SYRK_HPP
#define REMORA_KERNELS_HIP_SYRK_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <clblast.h>

#ifdef __NVCC__
#include "../../hip/cublas_backend.hpp"
#endif
namespace remora{ namespace kernels{

// C <- C + alpha * A * A^T
template <bool Upper, typename MatA, typename MatC>
void syrk(
	matrix_expression<MatA, hip_tag> const& A,
	matrix_expression<MatC, hip_tag>& C, 
	typename MatC::value_type const& alpha
) {
	REMORA_SIZE_CHECK(A().size1() == C().size1());
	REMORA_SIZE_CHECK(C().size1()== C().size2());
	
	static_assert(std::is_same<typename MatA::value_type, typename MatC::value_type>::value, "[syrk] Arguments do not have same element type");
	static_assert(std::is_same<typename MatA::evaluation_category::tag, dense_tag>::value, "[syrk] A is not dense");
	static_assert(std::is_base_of<dense_tag, typename MatC::storage_type::storage_tag>::value, "[syrk] C does not have dense storage layout");
	
	//pre-evaluate A into a temporary if necessary
	auto const& Aeval = eval_expression(A);
	
	//obtain geometry information
	bool transA = !std::is_same<typename MatA::orientation,typename MatC::orientation>::value;
	bool is_column_majorA = std::is_same<typename MatC::orientation::orientation, column_major>::value; 
	auto upperA = Upper; 
	if(!is_column_majorA){
		transA = !transA;
		upperA = !upperA;
	}
	
	
	std::size_t n = C().size1();
	std::size_t k = A().size2();

	//obtain matrix storage
	auto storageA = Aeval.raw_storage();
	auto storageC = C().raw_storage();
	
	hip::get_blas(C().queue()).syrk(
		upperA, transA,
		n, k, alpha,
		storageA.values, storageA.leading_dimension,
		typename MatC::value_type(1),
		storageC.values, storageC.leading_dimension,
		hip::get_stream(C().queue())
	);
}

}}

#endif
