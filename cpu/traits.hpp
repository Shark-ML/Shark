//===========================================================================
/*!
 * 
 *
 * \brief       Traits of matrix expressions
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
//===========================================================================

#ifndef REMORA_CPU_TRAITS_HPP
#define REMORA_CPU_TRAITS_HPP

#include "../expression_types.hpp"
#include "iterator.hpp"
#include <cmath>

namespace remora{

template<class Device>
struct device_traits;

template<>
struct device_traits<cpu_tag>{
	//queue (not used on cpu)
	struct queue_type{};
	
	static queue_type& default_queue(){
		static queue_type queue;
		return queue;
	}
	
	//adding of indices
	static std::size_t index_add(std::size_t i, std::size_t j){
		return i+j;
	}
	
	template<class E>
	static typename E::reference linearized_matrix_element(matrix_expression<E, cpu_tag> const& e, std::size_t i){
		std::size_t leading_dimension = E::orientation::index_m(e().size1(), e().size2());
		std::size_t i1 = i / leading_dimension;
		std::size_t i2 = i % leading_dimension;
		return e()(E::orientation::index_M(i1,i2), E::orientation::index_m(i1,i2));
	}
	
	template <class Iterator, class Functor>
	struct transform_iterator{
		typedef iterators::transform_iterator<Iterator, Functor> type;
	};
	
	template <class Iterator>
	struct subrange_iterator{
		typedef iterators::subrange_iterator<Iterator> type;
	};
	
	template <class Iterator1, class Iterator2, class Functor>
	struct binary_transform_iterator{
		typedef iterators::binary_transform_iterator<Iterator1,Iterator2, Functor> type;
	};
	
	template<class T>
	struct constant_iterator{
		typedef iterators::constant_iterator<T> type;
	};
	
	template<class T>
	struct one_hot_iterator{
		typedef iterators::one_hot_iterator<T> type;
	};
	
	template<class Closure>
	struct indexed_iterator{
		typedef iterators::indexed_iterator<Closure> type;
	};
	
	//functors
	template<class T>
	struct add {
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
	struct subtract {
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
	struct multiply {
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
	struct divide {
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
	struct multiply_and_add{
		static const bool left_zero_remains =  false;
		static const bool right_zero_remains =  false;
		static const bool right_zero_identity = true;
		static const bool left_zero_identity = false;
		typedef T result_type;
		multiply_and_add(T scalar):scalar(scalar){}
		T operator()(T x, T y)const{
			return x+scalar * y;
		}
	private:
		T scalar;
	};
	template<class T>
	struct multiply_assign{
		static const bool left_zero_remains =  false;
		static const bool right_zero_remains =  false;
		static const bool right_zero_identity = true;
		static const bool left_zero_identity = false;
		typedef T result_type;
		multiply_assign(T scalar):scalar(scalar){}
		T operator()(T, T y)const{
			return scalar * y;
		}
	private:
		T scalar;
	};
	template<class T>
	struct pow {
		static const bool left_zero_remains =  false;
		static const bool right_zero_remains =  false;
		typedef T result_type;
		T operator()(T x, T y)const {
			using std::pow;
			return pow(x,y);
		}
	};
	template<class T>
	struct multiply_scalar{
		static const bool zero_identity = true;
		typedef T result_type;
		multiply_scalar(T scalar):m_scalar(scalar){}
		T operator()(T x) const{
			return x * m_scalar;
		}
	private:
		T m_scalar;
	};
	template<class T>
	struct safe_divide {
		static const bool left_zero_remains =  true;
		static const bool right_zero_remains =  false;
		safe_divide(T defaultValue):m_defaultValue(defaultValue) {}
		typedef T result_type;
		T operator()(T x, T y)const{
			return y == T()? m_defaultValue : x/y;
		}
	private:
		T m_defaultValue;
	};
	
	//math unary functions
	#define REMORA_STD_UNARY_FUNCTION(func, id)\
	template<class T>\
	struct func{\
		static const bool zero_identity = id;\
		typedef T result_type;\
		T operator()(T x)const {\
			return std::func(x);\
		}\
	};

	REMORA_STD_UNARY_FUNCTION(abs, true)
	REMORA_STD_UNARY_FUNCTION(sqrt, true)
	REMORA_STD_UNARY_FUNCTION(cbrt, true)

	REMORA_STD_UNARY_FUNCTION(exp, false)
	REMORA_STD_UNARY_FUNCTION(log, false)

	//trigonometric functions
	REMORA_STD_UNARY_FUNCTION(cos, false)
	REMORA_STD_UNARY_FUNCTION(sin, true)
	REMORA_STD_UNARY_FUNCTION(tan, true)

	REMORA_STD_UNARY_FUNCTION(acos, false)
	REMORA_STD_UNARY_FUNCTION(asin, true)
	REMORA_STD_UNARY_FUNCTION(atan, true)

	//sigmoid type functions
	REMORA_STD_UNARY_FUNCTION(tanh, true)
	
	//special functions
	REMORA_STD_UNARY_FUNCTION(erf, false)
	REMORA_STD_UNARY_FUNCTION(erfc, false)
#undef REMORA_STD_UNARY_FUNCTION
	
	template<class T>
	struct sigmoid {
		static const bool zero_identity = false;
		typedef T result_type;
		T operator()(T x)const {
			using std::tanh;
			return (tanh(x/T(2)) + T(1))/T(2);
		}
	};
	template<class T>
	struct soft_plus {
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
	struct inv {
		static const bool zero_identity = false;
		typedef T result_type;	
		T operator()(T x)const {
			return T(1)/x;
		}
	};
	template<class T>
	struct sqr{
		static const bool zero_identity = true;
		typedef T result_type;
		T operator()(T x)const {
			return x*x;
		}
	};
	
	
	//min/max
	template<class T> 
	struct min{
		static const bool left_zero_remains =  false;
		static const bool right_zero_remains =  false;
		typedef T result_type;
		T operator()(T x, T y)const{
			return std::min(x,y);
		}
	};

	template<class T> 
	struct max{
		static const bool left_zero_remains =  false;
		static const bool right_zero_remains =  false;
		typedef T result_type;
		T operator()(T x, T y)const{
			return std::max(x,y);
		}
	};

	
	//comparison
	template<class T>
	struct less{
		static const bool left_zero_remains =  false;
		static const bool right_zero_remains =  false;
		typedef int result_type;
		int operator()(T x1, T x2)const {
			return x1 < x2;
		}
	};
	template<class T>
	struct less_equal{
		static const bool left_zero_remains =  false;
		static const bool right_zero_remains =  false;
		typedef int result_type;
		int operator()(T x1, T x2)const {
			return x1 <= x2;
		}
	};

	template<class T>
	struct bigger{
		static const bool left_zero_remains =  false;
		static const bool right_zero_remains =  false;
		typedef int result_type;
		int operator()(T x1, T x2)const {
			return x1 > x2;
		}
	};

	template<class T>
	struct bigger_equal{
		static const bool left_zero_remains =  false;
		static const bool right_zero_remains =  false;
		typedef int result_type;
		int operator()(T x1, T x2)const {
			return x1 >= x2;
		}
	};

	template<class T>
	struct equal{
		static const bool left_zero_remains =  false;
		static const bool right_zero_remains =  false;
		typedef int result_type;
		int operator()(T x1, T x2)const {
			return x1 ==  x2;
		}
	};

	template<class T>
	struct not_equal{
		static const bool left_zero_remains =  false;
		static const bool right_zero_remains =  false;
		typedef int result_type;
		int operator()(T x1, T x2)const {
			return x1 !=  x2;
		}
	};
};

}

#endif