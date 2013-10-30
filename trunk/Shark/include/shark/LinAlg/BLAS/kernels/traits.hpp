/*!
 *  \author O. Krause
 *  \date 2013
 *
 *  \par Copyright (c) 1998-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
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