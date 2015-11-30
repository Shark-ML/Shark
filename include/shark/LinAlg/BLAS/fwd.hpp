/*!
 * \brief      forward declarations
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
#ifndef SHARK_LINALG_BLAS_FWD_H
#define SHARK_LINALG_BLAS_FWD_H

namespace shark {
namespace blas {

// Storage types
struct range;

// Expression types

template<class E>
struct vector_expression;
template<class C>
struct vector_container;
template<class E>
class vector_reference;

template<class E>
struct matrix_expression;
template<class C>
struct matrix_container;
template<class E>
class matrix_reference;

template<class V>
class vector_range;

template<class M>
class matrix_row;
template<class M>
class matrix_column;
template<class M>
class matrix_vector_range;
template<class M>
class matrix_range;


// Sparse vectors
template<class T, class I = std::size_t>
class compressed_vector;

}
}

#endif
