/*!
 * 
 *
 * \brief       Generation of random variates
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
#ifndef REMORA_KERNELS_RANDOM_HPP
#define REMORA_KERNELS_RANDOM_HPP

#include "default/random.hpp"
#ifdef REMORA_USE_GPU
#include "gpu/random.hpp"
#endif
	
namespace remora{namespace kernels{
	
template<class V, class Rng, class Device>
void generate_normal(
	vector_expression<V, Device>& v,
	Rng& rng,
	typename V::value_type mean,
	typename V::value_type variance
) {
	bindings::generate_normal(v, rng, mean, variance);
}

template<class M, class Rng, class Device>
void generate_normal(
	matrix_expression<M, Device>& m,
	Rng& rng,
	typename M::value_type mean,
	typename M::value_type variance
) {
	bindings::generate_normal(m, rng, mean, variance);
}

template<class V, class Rng, class Device>
void generate_uniform(
	vector_expression<V, Device>& v,
	Rng& rng,
	typename V::value_type low,
	typename V::value_type high
) {
	bindings::generate_uniform(v, rng, low, high);
}

template<class M, class Rng, class Device>
void generate_uniform(
	matrix_expression<M, Device>& m,
	Rng& rng,
	typename M::value_type low,
	typename M::value_type high
) {
	bindings::generate_uniform(m, rng, low, high);
}

template<class V, class Rng, class Device>
void generate_discrete(
	vector_expression<V, Device>& v,
	Rng& rng,
	int low,
	int high
) {
	bindings::generate_discrete(v, rng, low, high);
}

template<class M, class Rng, class Device>
void generate_discrete(
	matrix_expression<M, Device>& m,
	Rng& rng,
	int low,
	int high
) {
	bindings::generate_discrete(m, rng, low, high);
}

}}
#endif