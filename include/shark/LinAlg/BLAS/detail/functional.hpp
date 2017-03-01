/*!
 * \brief       Functors used inside the library
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
#ifndef REMORA_DETAIL_FUNCTIONAL_HPP
#define REMORA_DETAIL_FUNCTIONAL_HPP

#include <cmath>

namespace remora {
namespace functors{
	
	
//////UNARY SCALAR OPRATIONS////////////////

template<class T>
struct scalar_inverse {
	static const bool zero_identity = false;
	typedef T result_type;	
	T operator()(T x)const {
		return T(1)/x;
	}
};

template<class T>
struct scalar_abs{
	static const bool zero_identity = true;
	typedef T result_type;
	T operator()(T x)const {
		using std::abs;
		return abs(x);
	}
};

template<class T>
struct scalar_sqr{
	static const bool zero_identity = true;
	typedef T result_type;
	T operator()(T x)const {
		return x*x;
	}
};

template<class T>
struct scalar_sqrt{
	static const bool zero_identity = true;
	typedef T result_type;
	T operator()(T x)const {
		using std::sqrt;
		return sqrt(x);
	}
};

template<class T>
struct scalar_exp{
	static const bool zero_identity = false;
	typedef T result_type;
	T operator()(T x)const {
		using std::exp;
		return exp(x);
	}
};

template<class T>
struct scalar_log {
	static const bool zero_identity = false;
	typedef T result_type;
	T operator()(T x)const {
		using std::log;
		return log(x);
	}
};

template<class T>
struct scalar_tanh{
	static const bool zero_identity = true;
	typedef T result_type;
	T operator()(T x)const {
		using std::tanh;
		return tanh(x);
	}
};

template<class T>
struct scalar_acos{
	static const bool zero_identity = false;
	typedef T result_type;
	T operator()(T x)const {
		using std::acos;
		return acos(x);
	}
};

template<class T>
struct scalar_soft_plus {
	static const bool zero_identity = false;
	typedef T result_type;
	T operator()(T x)const {
		if(x > 100){
			return x;
		}
		if(x < -100){
			return 0;
		}
		return std::log(1+std::exp(x));
	}
};

template<class T>
struct scalar_sigmoid {
	static const bool zero_identity = false;
	typedef T result_type;
	T operator()(T x)const {
		using std::tanh;
		return (tanh(x/T(2)) + T(1))/T(2);
	}
};

template<class T>
struct scalar_multiply1{
	static const bool zero_identity = true;
	typedef T result_type;
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
	typedef int result_type;
	int operator()(T x1, T x2)const {
		return x1 < x2;
	}
};
template<class T>
struct scalar_less_equal_than{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	typedef int result_type;
	int operator()(T x1, T x2)const {
		return x1 <= x2;
	}
};

template<class T>
struct scalar_bigger_than{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	typedef int result_type;
	int operator()(T x1, T x2)const {
		return x1 > x2;
	}
};

template<class T>
struct scalar_bigger_equal_than{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	typedef int result_type;
	int operator()(T x1, T x2)const {
		return x1 >= x2;
	}
};

template<class T>
struct scalar_equal{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	typedef int result_type;
	int operator()(T x1, T x2)const {
		return x1 ==  x2;
	}
};

template<class T>
struct scalar_not_equal{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	typedef int result_type;
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
	typedef T result_type;
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
	typedef T result_type;
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
	typedef T result_type;
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
	typedef T result_type;
	T operator()(T x, T y)const{
		return x/y;
	}
};

template<class T>
struct scalar_binary_safe_divide {
	static const bool left_zero_remains =  true;
	static const bool right_zero_remains =  false;
	scalar_binary_safe_divide(T defaultValue):m_defaultValue(defaultValue) {}
	typedef T result_type;
	T operator()(T x, T y)const{
		return y == T()? m_defaultValue : x/y;
	}
private:
	T m_defaultValue;
};

template<class T>
struct scalar_binary_multiply_and_add{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	static const bool right_zero_identity = true;
	static const bool left_zero_identity = false;
	typedef T result_type;
	scalar_binary_multiply_and_add(T scalar):scalar(scalar){}
	T operator()(T x, T y)const{
		return x+scalar * y;
	}
private:
	T scalar;
};

template<class T>
struct scalar_binary_multiply_assign{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	static const bool right_zero_identity = true;
	static const bool left_zero_identity = false;
	typedef T result_type;
	scalar_binary_multiply_assign(T scalar):scalar(scalar){}
	T operator()(T, T y)const{
		return scalar * y;
	}
private:
	T scalar;
};

template<class T>
struct scalar_binary_pow {
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	typedef T result_type;
	T operator()(T x, T y)const {
		using std::pow;
		return pow(x,y);
	}
};

template<class T> 
struct scalar_binary_min{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	typedef T result_type;
	T operator()(T x, T y)const{
		using std::min;
		return min(x,y);
	}
};

template<class T> 
struct scalar_binary_max{
	static const bool left_zero_remains =  false;
	static const bool right_zero_remains =  false;
	typedef T result_type;
	T operator()(T x, T y)const{
		using std::max;
		return max(x,y);
	}
};

}

}

#endif
