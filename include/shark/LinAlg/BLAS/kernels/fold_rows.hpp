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
template<class F, class M,class V, class Device>
void fold_rows(
	matrix_expression<M, Device> const & A, 
	vector_expression<V, Device>& b,
	F f,
	typename V::value_type alpha,
	unknown_orientation
){
	fold_rows(A, b, f, alpha, row_major());
}
}
	
namespace kernels{
///\brief Folds the rows of a row-major or column major matrix with a function f
///
/// output v_j is computed as v_j += alpha * f(A_0j, f(A_1j,...)))
/// Note: the implementation may assume that f is commutative and associative, i.e. the order of computation can be changed arbitrarily.
/// it is further assumed that if A only has 1 row, the result of just returning this value is correct
template <class F, class M, class V, class Device>
void fold_rows(
	matrix_expression<M, Device> const & A, 
	vector_expression<V, Device>& b,
	F f,
	typename V::value_type alpha
){
	REMORA_SIZE_CHECK(A().size2() == b().size());
	if(A().size1() == 0) return; //undefined
	bindings::fold_rows(
		A, b, f, alpha, typename M::orientation()
	);
}

}}

#endif
