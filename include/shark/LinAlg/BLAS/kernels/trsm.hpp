/*!
 * 
 *
 * \brief       Triangular solve kernel for matrix epressions.
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

#ifndef REMORA_KERNELS_TRSM_HPP
#define REMORA_KERNELS_TRSM_HPP

#include <type_traits> //std::false_type marker for unoptimized
#ifdef REMORA_USE_CBLAS
#include "cblas/trsm.hpp"
#else
// if no bindings are included, we have to provide the default has_optimized_gemv 
// otherwise the binding will take care of this
namespace remora{ namespace bindings{
template<class M1, class M2>
struct  has_optimized_trsm
: public std::false_type{};
}}
#endif

#include "default/trsm.hpp"

namespace remora{namespace kernels{
	
///\brief Implements the TRiangular Solver for Vectors.
///
/// It solves Systems of the form Ax = b where A is a square lower or upper triangular matrix.
/// It can optionally assume that the diagonal is 1 and won't access the diagonal elements.
template <class Triangular,class Side, typename MatA, typename MatB>
void trsm(
	matrix_expression<MatA, cpu_tag> const &A, 
	matrix_expression<MatB, cpu_tag> &B
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(!Side::is_left || A().size2() == B().size1());
	REMORA_SIZE_CHECK(Side::is_left || A().size2() == B().size2());
	
	bindings::trsm<Triangular, Side>(A,B,typename bindings::has_optimized_trsm<MatA, MatB>::type());
}

}}

#ifdef REMORA_USE_CLBLAST
#include "clBlast/trsm.hpp"
#elif defined REMORA_USE_GPU
#include "gpu/trsm.hpp"
#endif

#endif
