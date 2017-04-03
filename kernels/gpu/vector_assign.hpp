/*!
 * \brief       Assignment kernels for vector expressions
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
#ifndef REMORA_KERNELS_CLBLAS_VECTOR_ASSIGN_HPP
#define REMORA_KERNELS_CLBLAS_VECTOR_ASSIGN_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <boost/compute/iterator/zip_iterator.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/functional/bind.hpp>

namespace remora{namespace bindings{

template<class F, class V>
void assign(vector_expression<V, gpu_tag>& v, typename V::value_type t) {
	auto unary = boost::compute::bind(F(),boost::compute::placeholders::_1, t);
	boost::compute::transform(v().begin(),v().end(), v().begin(), unary, v().queue());
}

/////////////////////////////////////////////////////////
//direct assignment of two vectors
////////////////////////////////////////////////////////

// Dense-Dense case
template< class V, class E>
void vector_assign(
	vector_expression<V, gpu_tag>& v, vector_expression<E, gpu_tag> const& e, 
	dense_tag, dense_tag
) {
	boost::compute::copy(e().begin(),e().end(), v().begin(), v().queue());
}


////////////////////////////////////////////
//assignment with functor
////////////////////////////////////////////

// Dense-Dense case
template<class V, class E, class F>
void vector_assign_functor(
	vector_expression<V, gpu_tag>& v,
	vector_expression<E, gpu_tag> const& e,
	F f,
	dense_tag, dense_tag
) {
	auto zip_begin = boost::compute::make_zip_iterator(boost::make_tuple(v().begin(), e().begin()));
	auto zip_end = boost::compute::make_zip_iterator(boost::make_tuple(v().end(), e().end()));
	boost::compute::transform( zip_begin,zip_end, v().begin(), boost::compute::detail::unpack(f), v().queue());
}

}}
#endif
