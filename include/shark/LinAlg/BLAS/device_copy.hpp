/*!
 * \brief       expression templates for copying from cpu to device and back
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
#ifndef REMORA_DEVICE_COPY_HPP
#define REMORA_DEVICE_COPY_HPP

#include "expression_types.hpp"

namespace remora{

template<class E>
E const& copy_to_cpu(vector_expression<E, cpu_tag> const& e){
	return e();
}


template<class E>
E const&  copy_to_cpu(matrix_expression<E, cpu_tag> const& e){
	return e();
}

template<class E>
E const& copy_to_device(vector_expression<E, cpu_tag> const& e, cpu_tag){
	return e();
}


template<class E>
E const&  copy_to_device(matrix_expression<E, cpu_tag> const& e, cpu_tag){
	return e();
}

}

#ifdef REMORA_USE_GPU
#include "gpu/copy.hpp"
#endif

#endif
