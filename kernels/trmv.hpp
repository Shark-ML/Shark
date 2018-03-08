/*!
 * 
 *
 * \brief       Triangular matrix-vector multiplication kernel
 *
 * \author      O. Krause
 * \date        2012
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

#ifndef REMORA_KERNELS_TRMV_HPP
#define REMORA_KERNELS_TRMV_HPP

#ifdef REMORA_USE_CBLAS
#include "cblas/trmv.hpp"
#else
// if no bindings are included, we have to provide the default has_optimized_gemv 
// otherwise the binding will take care of this
namespace remora{ namespace bindings{
template<class M, class V>
struct  has_optimized_trmv
: public std::false_type{};
}}
#endif

#include "default/trmv.hpp"

namespace remora{namespace kernels{
	
///\brief Implements the TRiangular Solver for Vectors.
///
/// It solves Systems of the form Ax = b where A is a square lower or upper triangular matrix.
/// It can optionally assume that the diagonal is 1 and won't access the diagonal elements.
template <bool Upper,bool Unit,typename MatA, typename VecB>
void trmv(
	matrix_expression<MatA, cpu_tag> const &A, 
	vector_expression<VecB, cpu_tag>& b
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size1() == b().size());
	
	bindings::trmv<Upper,Unit>(A,b,typename bindings::has_optimized_trmv<MatA, VecB>::type());
}

}}

#ifdef REMORA_USE_CLBLAST
#include "clBlast/trmv.hpp"
#elif defined REMORA_USE_GPU
#include "gpu/trmv.hpp"
#endif

#endif
