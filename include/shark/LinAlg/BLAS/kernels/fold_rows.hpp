/*!
 * 
 *
 * \brief       Folds the rows of a row-major or column major matrix.
 *
 * \author      O. Krause
 * \date        2018
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

#ifndef REMORA_KERNELS_FOLD_ROWS_HPP
#define REMORA_KERNELS_FOLD_ROWS_HPP

#include "default/fold_rows.hpp"
#ifdef REMORA_USE_GPU
#include "gpu/fold_rows.hpp"
#endif

namespace remora {namespace bindings{
template<class F,  class G, class M,class V, class Device>
void fold_rows(
	matrix_expression<M, Device> const & A, 
	vector_expression<V, Device>& b,
	F f,
	G g,
	unknown_orientation
){
	fold_rows(A, b, f, g, row_major());
}
}
	
namespace kernels{
///\brief Folds each row of a matrix with a function f and transforms the result with another function g
///
/// output v_i is computed as v_i += g( f(A_i0, f(A_i1,... f(A_n-2i, A_n-1i) ))). That is, the result is the same
/// as folding each row separately as if it was a collection of numbers.
template <class F, class G, class M, class V, class Device>
void fold_rows(
	matrix_expression<M, Device> const & A, 
	vector_expression<V, Device>& b,
	F f,
	G g
){
	REMORA_SIZE_CHECK(A().size1() == b().size());
	if(A().size1() == 0) return; //undefined
	bindings::fold_rows(
		A, b, f, g, typename M::orientation()
	);
}

}}

#endif
