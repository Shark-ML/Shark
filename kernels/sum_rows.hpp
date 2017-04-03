/*!
 * 
 *
 * \brief       Sums the rows of a row-major or column major matrix.
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

#ifndef REMORA_KERNELS_SUM_ROWS_HPP
#define REMORA_KERNELS_SUM_ROWS_HPP

#include "default/sum_rows.hpp"
#ifdef REMORA_USE_GPU
#include "gpu/sum_rows.hpp"
#endif

namespace remora {namespace bindings{
template<class M,class V, class Device, class Tag1, class Tag2>
void sum_rows(
	matrix_expression<M, Device> const & A, 
	vector_expression<V, Device>& b,
	typename V::value_type alpha,
	unknown_orientation,
	Tag1, Tag2
){
	sum_rows(A,b,alpha,row_major(), Tag1(), Tag2());
}
}
	
namespace kernels{
///\brief Sums the rows of a row-major or column major matrix.
///
/// This is equivalent to the operation v=1^TA where 1 is the vector of all-ones
template <class M, class V, class Device>
void sum_rows(
	matrix_expression<M, Device> const & A, 
	vector_expression<V, Device>& b,
	typename V::value_type alpha
){
	SIZE_CHECK(A().size2() == b().size());
	
	bindings::sum_rows(A,b,alpha,typename M::orientation(),
	typename M::evaluation_category::tag(), typename V::evaluation_category::tag());
}

}}

#endif
