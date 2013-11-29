/*!
 *  \author O. Krause
 *  \date 2012
 *
 *  \par Copyright (c) 1998-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;result_type Bochum<BR>
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
#ifndef SHARK_LINALG_BLAS_UBLAS_KERNELS_DEFAULT_DOT_HPP
#define SHARK_LINALG_BLAS_UBLAS_KERNELS_DEFAULT_DOT_HPP

#include "../traits.hpp"
#include <boost/mpl/bool.hpp>

namespace shark { namespace blas {namespace bindings{

// Dense case
template<class E1, class E2, class result_type>
static void dot_impl(
	vector_expression<E1> const& v1,
	vector_expression<E2> const& v2,
	result_type& result,
	dense_random_access_iterator_tag,
	dense_random_access_iterator_tag
) {
	std::size_t size = v1().size();
	result = result_type();
	for(std::size_t i = 0; i != size; ++i){
		result += v1()(i) * v2()(i);
	}
}
// Sparse case
template<class E1, class E2, class result_type>
static void dot_impl(
	vector_expression<E1> const& v1,
	vector_expression<E2> const& v2,
	result_type& result,
	sparse_bidirectional_iterator_tag,
	sparse_bidirectional_iterator_tag
) {
	typename E1::const_iterator iter1=v1().begin();
	typename E1::const_iterator end1=v1().end();
	typename E2::const_iterator iter2=v2().begin();
	typename E2::const_iterator end2=v2().end();
	result = result_type();
	//be aware of empty vectors!
	while(iter1 != end1 && iter2 != end2)
	{
		std::size_t index1=iter1.index();
		std::size_t index2=iter2.index();
		if(index1==index2){
			result += *iter1 * *iter2;
			++iter1;
			++iter2;
		}
		else if(index1> index2){
			++iter2;
		}
		else {
			++iter1;
		}
	}
}

// Dense-Sparse case
template<class E1, class E2, class result_type>
static void dot_impl(
	vector_expression<E1> const& v1,
	vector_expression<E2> const& v2,
	result_type& result,
	dense_random_access_iterator_tag,
	sparse_bidirectional_iterator_tag
) {
	typename E2::const_iterator iter2=v2().begin();
	typename E2::const_iterator end2=v2().end();
	result = result_type();
	for(;iter2 != end2;++iter2){
		result += v1()(iter2.index()) * *iter2;
	}
}
//Sparse-Dense case is reduced to Dense-Sparse using symmetry.
template<class E1, class E2, class result_type>
static void dot_impl(
	vector_expression<E1> const& v1,
	vector_expression<E2> const& v2,
	result_type& result,
	sparse_bidirectional_iterator_tag t1,
	dense_random_access_iterator_tag t2
) {
	//use commutativity!
	dot_impl(v2,v1,result,t2,t1);
}
	
///\brief Implements the dot or inner product kernel s = x^Ty.
template<class E1, class E2,class result_type>
void dot(
	vector_expression<E1> const& v1,
	vector_expression<E2> const& v2,
	result_type& result,
	boost::mpl::false_
) {
	SIZE_CHECK(v1().size()==v2().size());
	return dot_impl(v1,v2,result,
		typename E1::const_iterator::iterator_category(),
		typename E2::const_iterator::iterator_category()
	);
}

}}}
#endif