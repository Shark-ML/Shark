/*!
 * \brief       Dense Matrix class
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
#ifndef REMORA_MATRIX_HPP
#define REMORA_MATRIX_HPP

#include "expression_types.hpp"
#include "detail/traits.hpp"
namespace remora {

/** \brief A dense matrix of values of type \c T.
 *
 * For a \f$(m \times n)\f$-dimensional matrix and \f$ 0 \leq i < m, 0 \leq j < n\f$, every element \f$ m_{i,j} \f$ is mapped to
 * the \f$(i.n + j)\f$-th element of the container for row major orientation or the \f$ (i + j.m) \f$-th element of
 * the container for column major orientation. In a dense matrix all elements are represented in memory in a
 * contiguous chunk of memory by definition.
 *
 * Orientation can also be specified, otherwise a \c row_major is used.
 *
 * \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
 * \tparam L the storage organization. It can be either \c row_major or \c column_major. Default is \c row_major
 */
template<class T, class L=row_major, class Tag = cpu_tag>
class matrix;
	

template<class T, class L, class Tag>
struct matrix_temporary_type<T,L,dense_tag, Tag>{
	typedef matrix<T,L, Tag> type;
};

template<class T, class Tag>
struct matrix_temporary_type<T,unknown_orientation,dense_tag, Tag>{
	typedef matrix<T,row_major, Tag> type;
};

}

#include "cpu/matrix.hpp"
#ifdef REMORA_USE_GPU
#include "gpu/matrix.hpp"
#endif

#endif
