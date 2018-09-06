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

#include "../detail/traits.hpp"
#include "../expression_types.hpp"
#include "iterator.hpp"
#include <cmath>



namespace remora{

template<>
struct device_traits<cpu_tag>{
	//queue (not used on cpu)
	typedef no_queue queue_type;
	
	static queue_type& default_queue(){
		static queue_type queue;
		return queue;
	}
	
	template <class Iterator, class Functor>
	struct transform_iterator{
		typedef iterators::transform_iterator<Iterator, Functor> type;
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

	//Arithmetic functors
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
	struct add_scalar{
		static const bool zero_identity = false;
		typedef T result_type;
		add_scalar(T scalar):m_scalar(scalar){}
		T operator()(T x) const{
			return x + m_scalar;
		}
	private:
		T m_scalar;
	};
	
	template<class T>
	struct divide_scalar{
		static const bool zero_identity = true;
		typedef T result_type;
		divide_scalar(T scalar):m_scalar(scalar){}
		T operator()(T x) const{
			return x / m_scalar;
		}
	private:
		T m_scalar;
	};
	
	template<class T>
	struct modulo_scalar{
		static const bool zero_identity = true;
		typedef T result_type;
		modulo_scalar(T scalar):m_scalar(scalar){}
		T operator()(T x) const{
			return x % m_scalar;
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
	struct greater{
		static const bool left_zero_remains =  false;
		static const bool right_zero_remains =  false;
		typedef int result_type;
		int operator()(T x1, T x2)const {
			return x1 > x2;
		}
	};

	template<class T>
	struct greater_equal{
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
	
	//special element structure
	template<class T>
	struct constant{
		typedef T result_type;
		constant(T const& value): m_value(value){}
		
		template<class Arg>
		T operator()(Arg const&) const{
			return m_value;
		}
		template<class Arg1, class Arg2>
		T operator()(Arg1 const&, Arg2 const&) const{
			return {m_value};
		}
		
		T m_value;
	};
	template<class T>
	struct unit{
		typedef T result_type;
		unit(T const& value, std::size_t index): m_value(value), m_index(index){}
		
		T operator()(std::size_t i) const{
			return (i == m_index)? m_value : T();
		}
		
		T m_value;
		std::size_t m_index;
	};
	
	template<class F>
	struct diag{
		typedef typename std::remove_reference<typename F::result_type>::type result_type;
		diag(F const& functor): m_functor(functor){}
		
		result_type operator()(std::size_t i, std::size_t j) const{
			return (i == j)? m_functor(i) : result_type();
		}
		
		F m_functor;
	};
	
	template<class T>
	struct vector_element{
		typedef T& result_type;
		vector_element(dense_vector_storage<T, dense_tag> const& storage):m_storage(storage){}
		
		T& operator()(std::size_t i) const{
			return m_storage.values[i * m_storage.stride];
		}
	private:
		dense_vector_storage<T, dense_tag> m_storage;
	};
	template<class T, class Orientation>
	struct matrix_element{
		typedef T& result_type;
		matrix_element(dense_matrix_storage<T, dense_tag> const& storage):m_storage(storage){}
		
		T& operator()(std::size_t i, std::size_t j) const{
			std::size_t stride1 = Orientation::stride1(m_storage.leading_dimension);
			std::size_t stride2 = Orientation::stride2(m_storage.leading_dimension);
			return m_storage.values[i * stride1 + j * stride2];
		}
	private:
		dense_matrix_storage<T, dense_tag> m_storage;
	};
	
	//functional
	
	template<class T>
	struct identity{
		typedef T result_type;
		
		T operator()(T arg) const{
			return arg;
		}
	};
	
	template<class T>
	struct left_arg{
		typedef T result_type;
		static const bool left_zero_remains =  true;
		static const bool right_zero_remains =  false;
		T operator()(T arg1, T) const{
			return arg1;
		}
	};
	template<class T>
	struct right_arg{
		typedef T result_type;
		static const bool left_zero_remains =  false;
		static const bool right_zero_remains =  true;
		T operator()(T, T arg2) const{
			return arg2;
		}
	};
	
	template<class F, class G>
	struct compose{
		typedef typename G::result_type result_type;
		compose(F const& f, G const& g): m_f(f), m_g(g){ }
		
		template<class Arg1>
		result_type operator()(Arg1 const& x) const{
			return m_g(m_f(x));
		}
		
		template<class Arg1, class Arg2>
		result_type operator()(Arg1 const& x, Arg2 const& y) const{
			return m_g(m_f(x,y));
		}
		
		F m_f;
		G m_g;
	};
	
	//G(F1(args),F2(args))
	template<class F1, class F2, class G>
	struct compose_binary{
		typedef typename G::result_type result_type;
		compose_binary(F1 const& f1, F2 const& f2, G const& g): m_f1(f1), m_f2(f2), m_g(g){ }
		
		template<class Arg1>
		result_type operator()( Arg1 const& x) const{
			return m_g(m_f1(x), m_f2(x));
		}
		template<class Arg1, class Arg2>
		result_type operator()( Arg1 const& x, Arg2 const& y) const{
			return m_g(m_f1(x,y), m_f2(x,y));
		}
		
		F1 m_f1;
		F2 m_f2;
		G m_g;
	};
	
	//G(F1(arg1),F2(arg2))
	template<class F1, class F2, class G>
	struct transform_arguments{
		typedef typename G::result_type result_type;
		transform_arguments(F1 const& f1, F2 const& f2, G const& g): m_f1(f1), m_f2(f2), m_g(g){ }
		
		template<class Arg1, class Arg2>
		result_type operator()( Arg1 const& x, Arg2 const& y) const{
			return m_g(m_f1(x),m_f2(y));
		}
		
		F1 m_f1;
		F2 m_f2;
		G m_g;
	};

	template<class F, class Arg2>
	struct bind_second{
		typedef typename F::result_type result_type;
		bind_second(F const& f, Arg2 const& arg2) : m_function(f), m_arg2(arg2){ }
		
		template<class Arg1>
		result_type operator()(Arg1 const& arg1) const{
			return m_function(arg1, m_arg2);
		}
		
		F m_function;
		Arg2 m_arg2;
		
	};
	
	//helper functions
	template<class F, class G>
	static compose<F,G> make_compose(F const& f, G const&g){
		return compose<F,G>(f,g);
	}
	
	template<class F1, class F2, class G>
	static compose_binary<F1, F2, G> make_compose_binary(F1 const& f1, F2 const& f2, G const&g){
		return compose_binary<F1, F2, G>(f1, f2, g);
	}
	
	template<class F1, class F2, class G>
	static transform_arguments<F1, F2, G> make_transform_arguments(F1 const& f1, F2 const& f2, G const& g){
		return transform_arguments<F1, F2, G>(f1, f2, g);
	}
	
	template<class F, class Arg2>
	static bind_second<F,Arg2> make_bind_second(F const& f, Arg2 const& arg2){
		return bind_second<F,Arg2>(f,arg2);
	}
	
};

}

#endif