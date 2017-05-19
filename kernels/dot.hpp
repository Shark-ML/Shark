/*!
 * 
 *
 * \brief       dot-product of vectors
 *
 * \author      O. Krause
 * \date        2013
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
#ifndef REMORA_KERNELS_DOT_HPP
#define REMORA_KERNELS_DOT_HPP

#include "default/dot.hpp"
#ifdef REMORA_USE_GPU
#include "gpu/dot.hpp"
#endif
	
namespace remora{namespace kernels{
	
///\brief Well known dot-product r=<e1,e2>=sum_i e1_i*e2_i.
///
/// If bindings are included and the vector combination allows for a specific binding
/// to be applied, the binding is called automatically from {binding}/dot.h
/// otherwise default/dot.h is used which is fully implemented for all dense/sparse combinations.
/// if a combination is optimized, bindings::has_optimized_dot<E1,E2,R>::type evaluates to std::true_type
/// The kernels themselves are implemented in bindings::dot.
template<class E1, class E2, class result_type, class Device>
void dot(
	vector_expression<E1, Device> const& e1,
	vector_expression<E2, Device> const& e2,
	result_type& result
) {
	REMORA_SIZE_CHECK(e1().size() == e2().size());
	
	bindings::dot(
		e1, e2,result,typename E1::evaluation_category::tag(), typename E2::evaluation_category::tag()
	);
}

}}
#endif