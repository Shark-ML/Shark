/*!
 * 
 *
 * \brief       Triangular Matrix-Matrix multiplication kernel
 *
 * \author      O. Krause
 * \date        2012
 *
 *
 * \par Copyright 1995-2014 Shark Developcbment Team
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

#ifndef REMORA_KERNELS_TRMM_HPP
#define REMORA_KERNELS_TRMM_HPP

#ifdef REMORA_USE_CBLAS
#include "cblas/trmm.hpp"
#else
// if no bindings are included, we have to provide the default has_optimized_gemv 
// otherwise the binding will take care of this
namespace remora{ namespace bindings{
template<class M1, class M2>
struct  has_optimized_trmm
: public std::false_type{};
}}
#endif

#include "default/trmm.hpp"

namespace remora{namespace kernels{
	
///\brief Implements the TRiangular Matrix Matrix multiply.
///
/// It computes B=A*B in place, where A is a triangular matrix and B a dense matrix
template <bool Upper,bool Unit,typename MatA, typename MatB>
void trmm(
	matrix_expression<MatA, cpu_tag> const &A, 
	matrix_expression<MatB, cpu_tag>& B
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	
	bindings::trmm<Upper,Unit>(A,B,typename bindings::has_optimized_trmm<MatA, MatB>::type());
}

}}

#ifdef REMORA_USE_CLBLAST
#include "clBlast/trmm.hpp"
#elif defined REMORA_USE_GPU
#include "gpu/trmm.hpp"
#endif

#endif
