/*!
 *  \brief basic math functions applied to whole vectors
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
#ifndef SHARK_LINALG_BLAS_VECTOR_TRANSFORMATIONS_H
#define SHARK_LINALG_BLAS_VECTOR_TRANSFORMATIONS_H

#include <shark/LinAlg/BLAS/Impl/UnaryExpressions.h>
#include <shark/LinAlg/BLAS/Impl/BinaryExpressions.h>
#include <shark/Core/Math.h>
namespace shark{
	
namespace detail{
	
///\brief Defines a unary function to use with Shark UnaryTransformation
///
///It can be used to create a functor called name which wraps a single argument function. The third arguments indicates whether
///f(0) = 0, which allows for certain optimization with sparse expressions
///Additionally a fourth argument opt can be added. it is used for adding a "using std::function" to enable
///Koenig Lookup for non standard types.
#define SHARK_UNARY_FUNCTOR(name,function,idZero,opt)\
struct name\
{\
	template<class T>\
	struct Functor{\
		typedef T value_type;\
		typedef T result_type;\
		static const bool zero_identity = idZero;\
\
		result_type operator()(const value_type& x)const{\
			opt;\
			return function(x);\
		}\
	};\
}

///\brief Defines a unary function to use with Shark UnaryTransformation. This version uses one argument in the constructor
///
///It can be used to create a functor called name which wraps a single argument function.
///Additionally a third argument opt can be added. it is used for adding a "using std::function" to enable
///Koenig Lookup for non standard types. the argument in the constructor is used as the second argument for function.

#define SHARK_UNARY_SCALAR_FUNCTOR(name,function,idZero,opt)\
template<class U>\
class name\
{\
public:\
	template<class T>\
	struct Functor{\
		typedef T result_type;\
		static const bool zero_identity = idZero;\
\
		Functor(U const& argument):m_argument(argument){}\
		result_type operator()(const T& x)const{\
			opt;\
			return function(x,m_argument);\
		}\
	private:\
		U m_argument;\
	};\
}

SHARK_UNARY_FUNCTOR(Exp,exp,false,using std::exp);
SHARK_UNARY_FUNCTOR(Log,log,false,using std::log);
SHARK_UNARY_FUNCTOR(TanH,tanh,true,using std::tanh);
SHARK_UNARY_FUNCTOR(Sigmoid,sigmoid,false,(void)0);
SHARK_UNARY_FUNCTOR(SoftPlus,softPlus,false,(void)0);
SHARK_UNARY_FUNCTOR(Sqr,sqr,true,(void)0);
SHARK_UNARY_FUNCTOR(Sqrt,sqrt,true,using std::sqrt);
SHARK_UNARY_FUNCTOR(Abs,abs,true,using std::abs);

SHARK_UNARY_SCALAR_FUNCTOR(Pow,pow,false,using std::pow);

#undef SHARK_UNARY_SCALAR_FUNCTOR
//later needed
//#undef SHARK_UNARY_FUNCTOR

template<class T>
struct SafeDivBinary{
	typedef T result_type;
	static const bool zero_identity = true;
	
	SafeDivBinary(T const& defaultValue):m_defaultValue(defaultValue){}
	template<class U>
	result_type operator()(U p, U q)const{
		if(q == U(0))
			return m_defaultValue;
		return p/q;
	}
private:
	T m_defaultValue;
};

}

///Uses a Functor to create a transformation which maps a vector on another vector where every element is transformed
///by the functor. The functor does not take any arguments
#define SHARK_UNARY_TRANSFORMATION(name, F)\
template<class E>\
typename boost::lazy_disable_if<\
	boost::is_arithmetic<E>,\
	detail::UnaryTransformation<E,F> \
>::type name(E const&e){\
	typedef typename detail::UnaryTransformation<E, F >::type type;\
	typedef typename detail::UnaryTransformation<E, F >::functor_type functor_type;\
	return type(e, functor_type());\
}
SHARK_UNARY_TRANSFORMATION(log, detail::Log)
SHARK_UNARY_TRANSFORMATION(exp, detail::Exp)
SHARK_UNARY_TRANSFORMATION(tanh,detail::TanH)
SHARK_UNARY_TRANSFORMATION(sigmoid, detail::Sigmoid)
SHARK_UNARY_TRANSFORMATION(softPlus, detail::SoftPlus)
SHARK_UNARY_TRANSFORMATION(sqr, detail::Sqr)
SHARK_UNARY_TRANSFORMATION(sqrt, detail::Sqrt)
SHARK_UNARY_TRANSFORMATION(abs, detail::Abs)
#undef SHARK_UNARY_TRANSFORMATION

///Uses a Functor to create a transformation which maps a vector on another vector where every element is transformed
///by the functor. The functor does take one argument which is the second argument to the function call
#define SHARK_UNARY_SCALAR_TRANSFORMATION(name, F)\
template<class E, class T>\
typename boost::lazy_disable_if<\
	boost::is_arithmetic<E>,\
	detail::UnaryTransformation<E,F<T> > \
>::type name(E const& e, T const& arg){\
	typedef typename detail::UnaryTransformation<E, F<T> >::type type;\
	typedef typename detail::UnaryTransformation<E, F<T> >::functor_type functor_type;\
	return type(e, functor_type(arg));\
}

SHARK_UNARY_SCALAR_TRANSFORMATION(pow, detail::Pow)
#undef SHARK_UNARY_SCALAR_TRANSFORMATION


///\brief Implements a safe division which checks for division by 0 and in this case returns
///       a default value.
template<class E1, class E2, class T>
typename detail::BinaryTransformation<
	E1,E2, 
	detail::SafeDivBinary<typename E1::value_type> 
>::type 
safeDiv(E1 const& e1, E2 const& e2, T const& defaultValue){
 	typedef detail::SafeDivBinary<typename E1::value_type> functor_type;
	typedef typename detail::BinaryTransformation<E1, E2, functor_type >::type type;
	return type(e1,e2,functor_type(defaultValue));
}

}

#endif
