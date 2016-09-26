/*!
 * \brief       Functors used inside the library
 * 
 * \author      O. Krause
 * \date        2013
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_LINALG_BLAS_DETAIL_FUNCTIONAL_HPP
#define SHARK_LINALG_BLAS_DETAIL_FUNCTIONAL_HPP

#include <boost/math/constants/constants.hpp>
#include "traits.hpp"
#include <shark/Core/Exception.h>
#include <shark/Core/Math.h>

namespace shark {
namespace blas {
namespace functors{
	
	
//////UNARY SCALAR OPRATIONS////////////////

template<class T>
struct scalar_negate {
	static const bool zero_identity = true;

	T operator()(T x)const {
		return -x;
	}
};

template<class T>
struct scalar_inverse {
	static const bool zero_identity = false;

	T operator()(T x)const {
		return T(1)/x;
	}
};

template<class T>
struct scalar_abs{
	static const bool zero_identity = true;

	T operator()(T x)const {
		using std::abs;
		return abs(x);
	}
};

template<class T>
struct scalar_sqr{
	static const bool zero_identity = true;

	T operator()(T x)const {
		return x*x;
	}
};

template<class T>
struct scalar_sqrt{
	static const bool zero_identity = true;

	T operator()(T x)const {
		using std::sqrt;
		return sqrt(x);
	}
};

template<class T>
struct scalar_exp{
	static const bool zero_identity = false;

	T operator()(T x)const {
		using std::exp;
		return exp(x);
	}
};

template<class T>
struct scalar_log {
	static const bool zero_identity = false;

	T operator()(T x)const {
		using std::log;
		return log(x);
	}
};

template<class T>
struct scalar_tanh{
	static const bool zero_identity = true;

	T operator()(T x)const {
		using std::tanh;
		return tanh(x);
	}
};

template<class T>
struct scalar_soft_plus {
	static const bool zero_identity = false;

	T operator()(T x)const {
		return shark::softPlus(x);
	}
};

template<class T>
struct scalar_sigmoid {
	static const bool zero_identity = false;

	T operator()(T x)const {
		using std::tanh;
		return (tanh(x/T(2)) + T(1))/T(2);
	}
};

template<class T>
struct scalar_multiply1{
	static const bool zero_identity = true;
	scalar_multiply1(T scalar):m_scalar(scalar){}
	T operator()(T x) const{
		return x * m_scalar;
	}
private:
	T m_scalar;
};

//////BINARY SCALAR OPRATIONS////////////////


template<class T>
struct scalar_less_than{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	int operator()(T x1, T x2)const {
		return x1 < x2;
	}
};
template<class T>
struct scalar_less_equal_than{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	int operator()(T x1, T x2)const {
		return x1 <= x2;
	}
};

template<class T>
struct scalar_bigger_than{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	int operator()(T x1, T x2)const {
		return x1 > x2;
	}
};

template<class T>
struct scalar_bigger_equal_than{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	int operator()(T x1, T x2)const {
		return x1 >= x2;
	}
};

template<class T>
struct scalar_equal{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	int operator()(T x1, T x2)const {
		return x1 ==  x2;
	}
};

template<class T>
struct scalar_not_equal{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	int operator()(T x1, T x2)const {
		return x1 !=  x2;
	}
};

template<class T>
struct scalar_binary_plus {
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	static const bool right_zero_identity = true;
	static const bool left_zero_identity = false;
	T operator()(T x, T y)const{
		return x+y;
	}
};
template<class T>
struct scalar_binary_minus {
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	static const bool right_zero_identity = true;
	static const bool left_zero_identity = false;
	T operator()(T x, T y)const{
		return x-y;
	}
};

template<class T>
struct scalar_binary_multiply {
	static const bool left_zero_remains =  true;
	static const bool right_zero_remains =  true;
	static const bool right_zero_identity = false;
	static const bool left_zero_identity = true;

	T operator()(T x, T y)const{
		return x*y;
	}
};

template<class T>
struct scalar_binary_divide {
	static const bool left_zero_remains =  true;
	static const bool right_zero_remains =  false;
	static const bool right_zero_identity = false;
	static const bool left_zero_identity = true;

	T operator()(T x, T y)const{
		return x/y;
	}
};

template<class T>
struct scalar_binary_safe_divide {
	static const bool left_zero_remains =  true;
	static const bool right_zero_remains =  false;
	scalar_binary_safe_divide(T defaultValue):m_defaultValue(defaultValue) {}

	T operator()(T x, T y)const{
		return y == T()? m_defaultValue : x/y;
	}
private:
	T m_defaultValue;
};

template<class T>
struct scalar_binary_pow {
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;

	T operator()(T x, T y)const {
		using std::pow;
		return pow(x,y);
	}
};

template<class T> 
struct scalar_binary_min{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;

	T operator()(T x, T y)const{
		using std::min;
		return min(x,y);
	}
};

template<class T> 
struct scalar_binary_max{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;

	T operator()(T x, T y)const{
		using std::max;
		return max(x,y);
	}
};

}

///////////////////VECTOR REDUCTION FUNCTORS/////////////////////////

//Functor implementing reduction of the form f(v_n,f(v_{n-1},f(....f(v_0,seed))))
// we assume for sparse vectors that the following holds:
// f(0,0) = 0 and f(v,f(0,w))=f(f(v,w),0)
//second argument to the function is the default value(seed).
template<class F>
struct vector_fold{
	
	vector_fold(F const& f):m_functor(f){}
	vector_fold(){}
	
	template<class E, class T>
	T operator()(
		vector_expression<E, cpu_tag> const& v,
		T seed
	) {
		return apply(v(),seed, typename E::evaluation_category::tag());
	}
private:
	//Dense Case
	template<class E, class T>
	T apply(
		E const& v,
		T seed,
		dense_tag
	) {
		std::size_t size = v.size();
		T result = seed;
		for(std::size_t i = 0; i != size; ++i){
			result = m_functor(result,v(i));
		}
		return result;
	}
	//Sparse Case
	template<class E, class T>
	T apply(
		E const& v,
		T seed,
		sparse_tag
	) {
		typename E::const_iterator iter=v.begin();
		typename E::const_iterator end=v.end();
		
		T result = seed;
		std::size_t nnz = 0;
		for(;iter != end;++iter,++nnz){
			result = m_functor(result,*iter);
		}
		//apply final operator f(0,v)
		if(nnz != v.size())
			result = m_functor(result,*iter);
		return result;
	}
	F m_functor;
};

}}

#endif
