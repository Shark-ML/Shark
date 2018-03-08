/*!
 * \brief       Algorithm to reduce a vector to a scalar value
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
#ifndef REMORA_KERNELS_MATRIX_FOLD_HPP
#define REMORA_KERNELS_MATRIX_FOLD_HPP

#include "../detail/traits.hpp"
#include "default/matrix_fold.hpp"
#ifdef REMORA_USE_GPU
#include "gpu/matrix_fold.hpp"
#endif

namespace remora{namespace kernels {


///\brief Applies F in any order to the elements of v and seed.
///
/// result is the same as value =f(v_1,f(v_2,...f(v_n,value))) assuming f is commutative
/// and associative.
template<class F, class M, class Device>
void matrix_fold(matrix_expression<M, Device> const& m, typename F::result_type& value) {
	typedef typename M::evaluation_category::tag Tag;
	typedef typename M::orientation Orientation;
	bindings::matrix_fold<F>(m, value, Orientation(), Tag());
}

}}
#endif
