/*!
 * 
 *
 * \brief       Kernel for calculating the maximum element of a vector
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
#ifndef REMORA_KERNELS_VECTOR_MAX_HPP
#define REMORA_KERNELS_VECTOR_MAX_HPP

#include "default/vector_max.hpp"
#ifdef REMORA_USE_GPU
#include "gpu/vector_max.hpp"
#endif
	
namespace remora { namespace kernels{
	
///\brief Computes the index of the maximum element of a vector
template<class E, class Device>
std::size_t vector_max(
	vector_expression<E, Device> const& e
) {
	REMORA_SIZE_CHECK(e().size() == e().size());
	return bindings::vector_max(e,typename E::evaluation_category::tag());
}

}}
#endif