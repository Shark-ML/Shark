/*!
 * 
 *
 * \brief       Triangular solve kernel for vector expressions.
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

#ifndef REMORA_KERNELS_TRSV_HPP
#define REMORA_KERNELS_TRSV_HPP

#include <type_traits>
#ifdef REMORA_USE_CBLAS
#include "cblas/trsv.hpp"
#else
// if no bindings are included, we have to provide the default has_optimized_gemv 
// otherwise the binding will take care of this
namespace remora {namespace bindings{
template<class M1, class M2>
struct  has_optimized_trsv
: public std::false_type{};
}}
#endif

#include "default/trsv.hpp"

namespace remora{namespace kernels{
	
///\brief Implements the TRiangular Solver for Vectors.
///
/// It solves Systems of the form Ax = b where A is a square lower or upper triangular matrix.
/// It can optionally assume that the diagonal is 1 and won't access the diagonal elements.
template <class Triangular, class Side, typename MatA, typename V>
void trsv(
	matrix_expression<MatA, cpu_tag> const &A, 
	vector_expression<V, cpu_tag> &b
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size1() == b().size());
	
	bindings::trsv<Triangular, Side>(A,b,typename bindings::has_optimized_trsv<MatA, V>::type());
}

}}

#ifdef REMORA_USE_CLBLAST
#include "clBlast/trsv.hpp"
#elif defined REMORA_USE_GPU
#include "gpu/trsv.hpp"
#endif

#endif
