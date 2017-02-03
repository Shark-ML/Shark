/*!
 * 
 *
 * \brief       -
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
#ifndef REMORA_KERNELS_CLBLAS_VECTOR_MAX_HPP
#define REMORA_KERNELS_CLBLAS_VECTOR_MAX_HPP

#include "../../detail/traits.hpp"
#include "../../expression_types.hpp"
#include <boost/compute/algorithm/max_element.hpp>
namespace remora {namespace bindings{

template<class E>
std::size_t vector_max(vector_expression<E, gpu_tag> const& v,dense_tag) {
	return static_cast<std::size_t>(boost::compute::max_element(v().begin(),v().end()) - v().begin());
}


}}
#endif