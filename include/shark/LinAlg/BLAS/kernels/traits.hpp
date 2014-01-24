/*!
 * 
 * \file        traits.hpp
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2013
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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

#ifndef SHARK_LINALG_BLAS_UBLAS_KERNELS_TRAITS_HPP
#define SHARK_LINALG_BLAS_UBLAS_KERNELS_TRAITS_HPP

#include "../traits.hpp"

namespace shark {namespace blas {namespace bindings{ namespace traits {
	
///////////////Vector Traits//////////////////////////
	
template <typename V>
typename V::difference_type stride(vector_expression<V> const&v) { 
	return v().stride();
}

template <typename V>
typename pointer<V>::type storage(vector_expression<V>& v) { 
	return v().storage();
}
template <typename V>
typename pointer<V const>::type storage(vector_expression<V> const& v) { 
	return v().storage();
}

//////////////////Matrix Traits/////////////////////
template <typename M>
typename M::difference_type stride1(matrix_expression<M> const& m) { 
	return m().stride1();
}
template <typename M>
typename M::difference_type stride2(matrix_expression<M> const& m) { 
	return m().stride2();
}

template <typename M>
typename pointer<M>::type storage(matrix_expression<M>& m) { 
	return m().storage();
}
template <typename M>
typename pointer<M const>::type storage(matrix_expression<M> const& m) { 
	return m().storage();
}

template <typename M>
typename M::difference_type leading_dimension(matrix_expression<M> const& m) {
	return  M::orientation::index_M(stride1(m),stride2(m));
}

template<class M1, class M2>
bool same_orientation(matrix_expression<M1> const& m1, matrix_expression<M2> const& m2){
	return boost::is_same<typename M1::orientation,typename M2::orientation>::value;
}


}}}}
#endif