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
#ifndef SHARK_LINALG_BLAS_UBLAS_KERNELS_DOT_HPP
#define SHARK_LINALG_BLAS_UBLAS_KERNELS_DOT_HPP

#include "default/dot.hpp"

#ifdef SHARK_USE_ATLAS
#include "atlas/dot.hpp"
#else
// if no bindings are included, we have to provide the default has_optimized_dot
// otherwise the binding will take care of this
namespace shark { namespace blas { namespace bindings{
template<class V1, class V2,class result_type>
struct  has_optimized_dot
: public boost::mpl::false_{};
}}}
#endif
	
namespace shark { namespace blas {namespace kernels{
	
///\brief Well known dot-product r=<e1,e2>=sum_i e1_i*e2_i.
///
/// If bindings are included and the vector combination allows for a specific binding
/// to be applied, the binding is called automatically from {binding}/dot.h
/// otherwise default/dot.h is used which is fully implemented for all dense/sparse combinations.
/// if a combination is optimized, bindings::has_optimized_dot<E1,E2,R>::type evaluates to boost::mpl::true_
/// The kernels themselves are implemented in blas::bindings::dot.
template<class E1, class E2,class result_type>
void dot(
	vector_expression<E1> const& e1,
	vector_expression<E2> const& e2,
	result_type& result
) {
	SIZE_CHECK(e1().size() == e2().size());
	
	bindings::dot(
		e1, e2,result,
		typename bindings::has_optimized_dot<E1,E2,result_type>::type()
	);
}

}}}
#endif