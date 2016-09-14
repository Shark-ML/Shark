/*!
 * 
 *
 * \brief       Vector proxy classes.
 * 
 * 
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

#ifndef SHARK_LINALG_BLAS_VECTOR_PROXY_HPP
#define SHARK_LINALG_BLAS_VECTOR_PROXY_HPP

#include "detail/expression_optimizers.hpp"


namespace shark{
namespace blas{

// ------------------
// Vector subrange
// ------------------

/// \brief Return a subrange of a specified vector, forming a vector for the specified indices between start and stop index.
///
/// The vector starts with first index being 0 for the element that is indexed with start in the original vector.
template<class V>
temporary_proxy<typename detail::vector_range_optimizer<V>::type>
subrange(vector_expression<V>& expression, typename V::size_type start, typename V::size_type stop){
	return detail::vector_range_optimizer<V>::create(expression(), start, stop);
}

template<class V>
typename detail::vector_range_optimizer<typename const_expression<V>::type>::type
subrange(vector_expression<V> const& expression, typename V::size_type start, typename V::size_type stop){
	return detail::vector_range_optimizer<typename const_expression<V>::type>::create(expression(), start, stop);
}

template<class V>
temporary_proxy<typename detail::vector_range_optimizer<V>::type>
subrange(temporary_proxy<V> expression, typename V::size_type start, typename V::size_type stop){
	return subrange(static_cast<V&>(expression), start, stop);
}

// ------------------
// Adapt memory as vector
// ------------------

/// \brief Converts a chunk of memory into a vector of a given size.
template <class T>
temporary_proxy<dense_vector_adaptor<T> > adapt_vector(std::size_t size, T * expression){
	return dense_vector_adaptor<T>(expression,size);
}

/// \brief Converts a C-style array into a vector.
template <class T, std::size_t N>
temporary_proxy<dense_vector_adaptor<T> > adapt_vector(T (&array)[N]){
	return dense_vector_adaptor<T>(array,N);
}


}
}

#endif
