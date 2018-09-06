//===========================================================================
/*!
 * 
 *
 * \brief       Traits of hip expressions
 *
 * \author      O. Krause
 * \date        2018
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

#ifndef REMORA_HIP_TRAITS_HPP
#define REMORA_HIP_TRAITS_HPP

#include "../detail/traits.hpp"
#include "device.hpp"

namespace remora{
template<>
struct device_traits<hip_tag>{
	typedef hip::device queue_type;
	
	static queue_type& default_queue(){
		return hip::devices().device(0);
	}
	
	// iterators
	
	template <class Iterator, class Functor>
	struct transform_iterator{
		typedef no_iterator type;
	};
	
	template <class Iterator1, class Iterator2, class Functor>
	struct binary_transform_iterator{
		typedef no_iterator type;
	};
	
	template<class T>
	struct constant_iterator{
		typedef no_iterator type;
	};
	
	template<class T>
	struct one_hot_iterator{
		typedef no_iterator type;
	};
	
	template<class Closure>
	struct indexed_iterator{
		typedef no_iterator type;
	};
	
	//functors
	
#define REMORA_BINARY_FUNCTION(func, op, R)\
	template<class T>\
	struct func{\
		typedef R result_type;\
		__device__ T operator()(T x, T y)const {\
			return op;\
		}\
	};
	
	//arithmetic
	REMORA_BINARY_FUNCTION(add, x+y, T);
	REMORA_BINARY_FUNCTION(subtract, x-y, T);
	REMORA_BINARY_FUNCTION(multiply, x*y, T);
	REMORA_BINARY_FUNCTION(divide, x/y, T);
	//binary functions
	REMORA_BINARY_FUNCTION(pow, ::pow(x,y), T);
	REMORA_BINARY_FUNCTION(min, ::min(x,y), T);
	REMORA_BINARY_FUNCTION(max, ::max(x,y), T);
	//comparison
	REMORA_BINARY_FUNCTION(less, x<y, int);
	REMORA_BINARY_FUNCTION(greater, x>y, int);
	REMORA_BINARY_FUNCTION(less_equal, x<=y, int);
	REMORA_BINARY_FUNCTION(greater_equal, x>=y, int);
	REMORA_BINARY_FUNCTION(equal, x==y, int);
	REMORA_BINARY_FUNCTION(not_equal, x!=y, int);
#undef REMORA_BINARY_FUNCTION
	
#define REMORA_UNARY_SCALAR_FUNCTION(func, call)\
	template<class T>\
	struct func{\
		typedef T result_type;\
		func(T scalar):m_scalar(scalar){}\
		__device__ T operator()(T x) const{\
			return call;\
		}\
	private:\
		T m_scalar;\
	};
	
	REMORA_UNARY_SCALAR_FUNCTION(add_scalar,x + m_scalar);
	REMORA_UNARY_SCALAR_FUNCTION(multiply_scalar, x * m_scalar);
	REMORA_UNARY_SCALAR_FUNCTION(divide_scalar, x / m_scalar);
	REMORA_UNARY_SCALAR_FUNCTION(modulo_scalar, x % m_scalar);
	
#undef REMORA_UNARY_SCALAR_FUNCTION
	
	template<class T>
	struct multiply_and_add{
		typedef T result_type;
		multiply_and_add(T scalar):scalar(scalar){}
		__device__ T operator()(T x, T y)const{
			return x + scalar * y;
		}
	private:
		T scalar;
	};
	template<class T>
	struct multiply_assign{
		typedef T result_type;
		multiply_assign(T scalar):scalar(scalar){}
		__device__ T operator()(T, T y)const{
			return scalar * y;
		}
	private:
		T scalar;
	};
	
	
	template<class T>
	struct safe_divide {
		typedef T result_type;
		safe_divide(T defaultValue):m_defaultValue(defaultValue) {}
		__device__ T operator()(T x, T y)const{
			return y == T()? m_defaultValue : x/y;
		}
	private:
		T m_defaultValue;
	};
	
	//math unary functions
#define REMORA_STD_UNARY_FUNCTION(func)\
	template<class T>\
	struct func{\
		typedef T result_type;\
		__device__ T operator()(T x)const {\
			return ::func(x);\
		}\
	};

	REMORA_STD_UNARY_FUNCTION(abs)
	REMORA_STD_UNARY_FUNCTION(sqrt)
	REMORA_STD_UNARY_FUNCTION(cbrt)

	REMORA_STD_UNARY_FUNCTION(exp)
	REMORA_STD_UNARY_FUNCTION(log)

	//trigonometric functions
	REMORA_STD_UNARY_FUNCTION(cos)
	REMORA_STD_UNARY_FUNCTION(sin)
	REMORA_STD_UNARY_FUNCTION(tan)

	REMORA_STD_UNARY_FUNCTION(acos)
	REMORA_STD_UNARY_FUNCTION(asin)
	REMORA_STD_UNARY_FUNCTION(atan)

	//sigmoid type functions
	REMORA_STD_UNARY_FUNCTION(tanh)
	
	//special functions
	REMORA_STD_UNARY_FUNCTION(erf)
	REMORA_STD_UNARY_FUNCTION(erfc)
#undef REMORA_STD_UNARY_FUNCTION
	
	template<class T>
	struct sigmoid{
		typedef T result_type;
		__device__ T operator()(T x)const {
			return (std::tanh(x/T(2)) + T(1))/T(2);
		}
	};
	template<class T>
	struct soft_plus{
		typedef T result_type;
		__device__ T operator()(T x)const {
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
		typedef T result_type;
		__device__ T operator()(T x)const {
			return T(1)/x;
		}
	};
	template<class T>
	struct sqr{
		typedef T result_type;
		__device__ T operator()(T x)const {
			return x*x;
		}
	};
	
	//special element structure
	template<class T>
	struct constant{
		typedef T result_type;
		constant(T const& value): m_value(value){}
		
		template<class Arg>
		__device__ T operator()(Arg const&) const{
			return m_value;
		}
		template<class Arg1, class Arg2>
		__device__ T operator()(Arg1 const&, Arg2 const&) const{
			return m_value;
		}
		
		T m_value;
	};
	template<class T>
	struct unit{
		typedef T result_type;
		unit(T const& value, std::size_t index): m_value(value), m_index(index){}
		
		__device__ T operator()(std::size_t i) const{
			return (i == m_index)? m_value : T();
		}
		
		T m_value;
		std::size_t m_index;
	};
	
	template<class F>
	struct diag{
		typedef typename std::remove_reference<typename F::result_type>::type result_type;
		diag(F const& functor): m_functor(functor){}
		
		__device__ result_type operator()(std::size_t i, std::size_t j) const{
			return (i == j)? m_functor(i) : result_type();
		}
		
		F m_functor;
	};
	
	template<class T>
	struct vector_element{
		typedef T& result_type;
		vector_element(dense_vector_storage<T, dense_tag> const& storage):m_storage(storage){}
		
		__device__ T& operator()(std::size_t i) const{
			return m_storage.values[i * m_storage.stride];
		}
	private:
		dense_vector_storage<T, dense_tag> m_storage;
	};
	template<class T, class Orientation>
	struct matrix_element{
		typedef T& result_type;
		matrix_element(dense_matrix_storage<T, dense_tag> const& storage):m_storage(storage){}
		
		__device__ T& operator()(std::size_t i, std::size_t j) const{
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
		__device__ T operator()(T arg) const{
			return arg;
		}
	};
	
	template<class T>
	struct left_arg{
		typedef T result_type;
		__device__ T operator()(T arg1, T) const{
			return arg1;
		}
	};
	template<class T>
	struct right_arg{
		typedef T result_type;
		__device__ T operator()(T, T arg2) const{
			return arg2;
		}
	};
	
	template<class F, class G>
	struct compose{
		typedef typename G::result_type result_type;
		compose(F const& f, G const& g): m_f(f), m_g(g){ }
		
		template<class Arg1>
		__device__ result_type operator()(Arg1 const& x) const{
			return m_g(m_f(x));
		}
		
		template<class Arg1, class Arg2>
		__device__ result_type operator()(Arg1 const& x, Arg2 const& y) const{
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
		__device__ result_type operator()( Arg1 const& x) const{
			return m_g(m_f1(x), m_f2(x));
		}
		template<class Arg1, class Arg2>
		__device__ result_type operator()( Arg1 const& x, Arg2 const& y) const{
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
		__device__ result_type operator()( Arg1 const& x, Arg2 const& y) const{
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
		__device__ result_type operator()(Arg1 const& arg1) const{
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