//===========================================================================
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
//===========================================================================
#ifndef SHARK_LINALG_BLAS_KERNELS_CLBLAS_DOT_HPP
#define SHARK_LINALG_BLAS_KERNELS_CLBLAS_DOT_HPP

#include "clblas_inc.hpp"
#include <boost/compute/algorithm/inner_product.hpp>

namespace shark {namespace blas {namespace bindings {

template <typename VectorX, typename VectorY>
void dot(
	vector_expression<VectorX, gpu_tag> const& x,
        vector_expression<VectorY, gpu_tag> const& y,
	typename VectorX::value_type& result,
	dense_tag,
	dense_tag
){
	result = boost::compute::inner_product(x().begin(),x().end(),y().begin(), typename VectorX::value_type(0), x().queue());
}

}

namespace kernels{
template <typename VectorX, typename VectorY>
void dot(
	vector_expression<VectorX, gpu_tag> const& x,
        vector_expression<VectorY, gpu_tag> const& y,
	typename VectorX::value_type& result
){
	SIZE_CHECK(x().size() == y().size());

	bindings::dot(x,y, result, typename VectorX::evaluation_category::tag(), typename VectorY::evaluation_category::tag());
}

}}}
#endif
