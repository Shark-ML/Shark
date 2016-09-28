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
#ifndef SHARK_LINALG_BLAS_KERNELS_VECTOR_FOLD_HPP
#define SHARK_LINALG_BLAS_KERNELS_VECTOR_FOLD_HPP

#include "../detail/traits.hpp"
#include "default/vector_fold.hpp"
#ifdef SHARK_USE_CLBLAS
#include "clblas/vector_fold.hpp"
#endif

namespace shark {namespace blas {namespace kernels {


///\brief Appliues F in any order to the elements of v and seed.
///
/// result is the same as f(v_1,f(v_2,...f(v_n,seed))) assuming f is commutative
/// and associative.
template<class F, class V, class Device>
typename F::result_type vector_fold(vector_expression<V, Device> const& v, typename F::result_type seed) {
	typedef typename V::evaluation_category::tag TagV;
	typename F::result_type value = seed;
	bindings::vector_fold<F>(v(), value, TagV());
	return value;
}

}}}
#endif
