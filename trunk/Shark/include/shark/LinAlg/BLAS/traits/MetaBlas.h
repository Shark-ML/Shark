/*!
 *  \brief Meta template transformations and types for generic programming with ublas.
 *
 *  \author O.Krause
 *  \date 2010-2011
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
#ifndef SHARK_LINALG_BLAS_TRAITS_METABLAS_H
#define SHARK_LINALG_BLAS_TRAITS_METABLAS_H

#include <shark/LinAlg/BLAS/ublas.h>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/find_if.hpp>

namespace shark{
namespace detail{
/// \brief  If T is an expression template and has a base of type expression<T>, and if this base is supported, ExpressionType<T>::type returns expression<T>. Currently only the ublas expression types are supported.
template<class T>
struct ExpressionType{
private:
	///vector of supported expression types
	typedef boost::mpl::vector<
		boost::numeric::ublas::vector_expression<T>,
		boost::numeric::ublas::matrix_expression<T>,
		boost::numeric::ublas::scalar_expression<T>
		//add more here if needed
	> supportedExpressions;
	
	//either T is derived from one of the Expressions defined above, or T is the exact type.
	//so find the expression for this it is fulfilled.
	typedef typename boost::mpl::find_if<
		supportedExpressions,
		boost::mpl::or_<boost::is_base_of<boost::mpl::_1, T>, boost::is_same<boost::mpl::_1, T> >
	>::type pos;
public:
	typedef typename boost::mpl::deref<pos>::type type;
	typedef T underlying;
};
//specialization for the case that T already is an expression...
//todo: find a better way to do that
template<class T>
struct ExpressionType<boost::numeric::ublas::matrix_expression<T> >{
	typedef boost::numeric::ublas::matrix_expression<T> type;
	typedef T underlying;
};
template<class T>
struct ExpressionType<boost::numeric::ublas::vector_expression<T> >{
	typedef boost::numeric::ublas::vector_expression<T> type;
	typedef T underlying;
};
}
}

#endif
