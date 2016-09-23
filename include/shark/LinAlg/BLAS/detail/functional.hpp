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

struct scalar_negate {
	static const bool zero_identity = true;

	template<class T>
	T operator()(T x)const {
		return -x;
	}
};

struct scalar_inverse {
	static const bool zero_identity = false;

	template<class T>
	T operator()(T x)const {
		return T(1)/x;
	}
};


struct scalar_abs{
	static const bool zero_identity = true;

	template<class T>
	T operator()(T x)const {
		using std::abs;
		return abs(x);
	}
};

struct scalar_sqr{
	static const bool zero_identity = true;

	template<class T>
	T operator()(T x)const {
		return x*x;
	}
};

struct scalar_sqrt{
	static const bool zero_identity = true;

	template<class T>
	T operator()(T x)const {
		using std::sqrt;
		return sqrt(x);
	}
};

struct scalar_exp{
	static const bool zero_identity = false;

	template<class T>
	T operator()(T x)const {
		using std::exp;
		return exp(x);
	}
};

struct scalar_log {
	static const bool zero_identity = false;

	template<class T>
	T operator()(T x)const {
		using std::log;
		return log(x);
	}
};

struct scalar_tanh{
	static const bool zero_identity = true;

	template<class T>
	T operator()(T x)const {
		using std::tanh;
		return tanh(x);
	}
};

struct scalar_soft_plus {
	static const bool zero_identity = false;

	template<class T>
	T operator()(T x)const {
		return shark::softPlus(x);
	}
};

struct scalar_sigmoid {
	static const bool zero_identity = false;

	template<class T>
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


struct scalar_less_than{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	template<class T1, class T2>
	int operator()(T1 x1, T2 x2)const {
		return x1 < x2;
	}
};

struct scalar_less_equal_than{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	template<class T1, class T2>
	int operator()(T1 x1, T2 x2)const {
		return x1 <= x2;
	}
};

struct scalar_bigger_than{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	template<class T1, class T2>
	int operator()(T1 x1, T2 x2)const {
		return x1 > x2;
	}
};

struct scalar_bigger_equal_than{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	template<class T1, class T2>
	int operator()(T1 x1, T2 x2)const {
		return x1 >= x2;
	}
};

struct scalar_equal{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	template<class T1, class T2>
	int operator()(T1 x1, T2 x2)const {
		return x1 ==  x2;
	}
};

struct scalar_not_equal{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	template<class T1, class T2>
	int operator()(T1 x1, T2 x2)const {
		return x1 !=  x2;
	}
};

struct scalar_binary_plus {
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	static const bool right_zero_identity = true;
	static const bool left_zero_identity = false;
	template<class T1, class T2>
	auto operator()(T1 x, T2 y)const -> decltype(T1() + T2()) {
		return x+y;
	}
};
struct scalar_binary_minus {
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	static const bool right_zero_identity = true;
	static const bool left_zero_identity = false;
	template<class T1, class T2>
	auto operator()(T1 x, T2 y)const -> decltype(T1() + T2()) {
		return x-y;
	}
};

struct scalar_binary_multiply {
	static const bool left_zero_remains =  true;
	static const bool right_zero_remains =  true;
	static const bool right_zero_identity = false;
	static const bool left_zero_identity = true;
	template<class T1, class T2>
	auto operator()(T1 x, T2 y)const -> decltype(T1() * T2()) {
		return x*y;
	}
};

struct scalar_binary_divide {
	static const bool left_zero_remains =  true;
	static const bool right_zero_remains =  false;
	static const bool right_zero_identity = false;
	static const bool left_zero_identity = true;
	template<class T1, class T2>
	auto operator()(T1 x, T2 y)const -> decltype(T1() / T2()) {
		return x/y;
	}
};

template<class T>
struct scalar_binary_safe_divide {
	static const bool left_zero_remains =  true;
	static const bool right_zero_remains =  false;
	scalar_binary_safe_divide(T defaultValue):m_defaultValue(defaultValue) {}
	template<class T1, class T2>
	auto operator()(T1 x, T2 y)const -> decltype(T1() / T2()) {
		typedef decltype(T1() / T2()) result_type;
		return y == T2()? static_cast<result_type>(m_defaultValue) : x/y;
	}
private:
	T m_defaultValue;
};

struct scalar_binary_pow {
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	template<class T1, class T2>
	auto operator()(T1 x, T2 y)const -> decltype(T1() * T2()) {
		using std::pow;
		return pow(x,y);
	}
};

struct scalar_binary_min{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	template<class T1, class T2>
	auto operator()(T1 x, T2 y)const->decltype(T1() + T2()) {
		typedef decltype(T1() + T2()) result_type;
		using std::min;
		//convert to the bigger type to prevent std::min conversion errors.
		return min(result_type(x),result_type(y));
	}
};

struct scalar_binary_max{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	template<class T1, class T2>
	auto operator()(T1 x, T2 y)const->decltype(T1() + T2()) {
		typedef decltype(T1() + T2()) result_type;
		using std::max;
		//convert to the bigger type to prevent std::max conversion errors.
		return max(result_type(x),result_type(y));
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
