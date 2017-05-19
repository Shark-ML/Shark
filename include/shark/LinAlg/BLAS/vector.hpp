/*!
 * \brief       Implements the Dense vector class
 * 
 * \author      O. Krause
 * \date        2014
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
#ifndef REMORA_VECTOR_HPP
#define REMORA_VECTOR_HPP

#include "expression_types.hpp"
#include "detail/traits.hpp"
namespace remora{
	
/// \brief A dense vector of values of type \c T.
///
/// For a \f$n\f$-dimensional vector \f$v\f$ and \f$0\leq i < n\f$ every element \f$v_i\f$ is mapped
/// to the \f$i\f$-th element of the container.
/// The tag descripes whether the vector is residing on a cpu or gpu which change its semantics.
template<class T, class Tag = cpu_tag>
class vector;

template<class T, class Tag>
struct vector_temporary_type<T,dense_tag, Tag>{
	typedef vector<T, Tag> type;
};
}

#include "cpu/vector.hpp"
#ifdef REMORA_USE_GPU
#include "gpu/vector.hpp"
#endif

#endif
