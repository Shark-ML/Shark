/*!
 * 
 *
 * \brief       Traits to obtain memory and storage sizes easily from expression templates
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

#ifndef SHARK_LINALG_BLAS_KERNELS_TRAITS_HPP
#define SHARK_LINALG_BLAS_KERNELS_TRAITS_HPP

#include "../detail/traits.hpp"

namespace shark {namespace blas {namespace bindings{ namespace traits {

template<class M1, class M2>
bool same_orientation(matrix_expression<M1> const& m1, matrix_expression<M2> const& m2){
	return boost::is_same<typename M1::orientation,typename M2::orientation>::value;
}


}}}}
#endif