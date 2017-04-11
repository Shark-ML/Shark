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

#ifndef REMORA_DETAIL_TRAITS_HPP
#define REMORA_DETAIL_TRAITS_HPP

#include "iterator.hpp"
#include "evaluation_tags.hpp"
#include "structure.hpp"
#include "storage.hpp"
#include "../expression_types.hpp"

#include <complex>
#include <type_traits>

namespace remora {
	
template<class T>
struct real_traits{
	typedef T type;
};

template<class T>
struct real_traits<std::complex<T> >{
	typedef T type;
};

template<class E>
struct closure: public std::conditional<
	std::is_const<E>::value,
	typename E::const_closure_type,
	typename E::closure_type
>{};
	
template<class E>
struct const_expression : public std::conditional<
	std::is_base_of<vector_container<typename std::remove_const<E>::type,typename E::device_type>, E >::value
	||std::is_base_of<matrix_container<typename std::remove_const<E>::type,typename E::device_type>, E >::value,
	E const,
	typename E::const_closure_type
>{};

template<class E>
struct reference: public std::conditional<
	std::is_const<E>::value,
	typename E::const_reference,
	typename E::reference
>{};

template<class E>
struct storage: public std::conditional<
	std::is_const<E>::value,
	typename E::const_storage_type,
	typename E::storage_type
>{};
	
template<class M>
struct row_iterator: public std::conditional<
	std::is_const<M>::value,
	typename M::const_row_iterator,
	typename M::row_iterator
>{};
	
template<class M>
struct column_iterator: public std::conditional<
	std::is_const<M>::value,
	typename M::const_column_iterator,
	typename M::column_iterator
>{};

template<class Matrix> 
struct major_iterator:public std::conditional<
	std::is_same<typename Matrix::orientation, column_major>::value,
	typename column_iterator<Matrix>::type,
	typename row_iterator<Matrix>::type
>{};	
	
namespace detail{
	template<class M>
	typename column_iterator<M>::type major_begin(M& m,std::size_t i, column_major){
		return m.column_begin(i);
	}
	template<class M>
	typename row_iterator<M>::type major_begin(M& m,std::size_t i, row_major){
		return m.row_begin(i);
	}
	template<class M>
	typename row_iterator<M>::type major_begin(M& m,std::size_t i, unknown_orientation){
		return m.row_begin(i);
	}
	template<class M>
	typename column_iterator<M>::type major_end(M& m,std::size_t i, column_major){
		return m.column_end(i);
	}
	template<class M>
	typename row_iterator<M>::type major_end(M& m,std::size_t i, row_major){
		return m.row_end(i);
	}
	template<class M>
	typename row_iterator<M>::type major_end(M& m,std::size_t i, unknown_orientation){
		return m.row_end(i);
	}
}

template<class M, class Device>
typename major_iterator<M const>::type major_begin(matrix_expression<M, Device> const& m, std::size_t i){
	return detail::major_begin(m(),i, typename M::orientation::orientation());
}
template<class M, class Device>
typename major_iterator<M const>::type major_end(matrix_expression<M, Device> const& m, std::size_t i){
	return detail::major_end(m(),i, typename M::orientation::orientation());
}
template<class M, class Device>
typename major_iterator<M>::type major_begin(matrix_expression<M, Device>& m, std::size_t i){
	return detail::major_begin(m(),i, typename M::orientation::orientation());
}
template<class M, class Device>
typename major_iterator<M>::type major_end(matrix_expression<M, Device>& m, std::size_t i){
	return detail::major_end(m(),i, typename M::orientation::orientation());
}

///\brief Determines a good vector type storing an expression returning values of type T having a certain evaluation category on a specific device.
template<class ValueType, class Cateogry, class Device>
struct vector_temporary_type;
///\brief Determines a good vector type storing an expression returning values of type T having a certain evaluation category on a specific device.
template<class ValueType, class Orientation, class Category, class Device>
struct matrix_temporary_type;

/// For the creation of temporary vectors in the assignment of proxies
template <class E>
struct vector_temporary{
	typedef typename vector_temporary_type<
		typename E::value_type,
		typename E::evaluation_category::tag,
		typename E::device_type
	>::type type;
};

/// For the creation of temporary matrix in the assignment of proxies
template <class E>
struct matrix_temporary{
	typedef typename matrix_temporary_type<
		typename E::value_type,
		typename E::orientation,
		typename E::evaluation_category::tag,
		typename E::device_type
	>::type type;
};

/// For the creation of transposed temporary matrix in the assignment of proxies
template <class E>
struct transposed_matrix_temporary{
	typedef typename matrix_temporary_type<
		typename E::value_type,
		typename E::orientation::transposed_orientation,
		typename E::evaluation_category::tag,
		typename E::device_type
	>::type type;
};

namespace detail{
	template<class Matrix, class Device>
	void ensure_size(matrix_expression<Matrix, Device>& mat,std::size_t rows, std::size_t columns){
		SIZE_CHECK(mat().size1() == rows);
		SIZE_CHECK(mat().size2() == columns);
	}
	template<class Matrix, class Device>
	void ensure_size(matrix_container<Matrix, Device>& mat,std::size_t rows, std::size_t columns){
		mat().resize(rows,columns);
	}
	template<class Vector, class Device>
	void ensure_size(vector_expression<Vector, Device>& vec,std::size_t size){
		SIZE_CHECK(vec().size() == size);
	}
	template<class Vector, class Device>
	void ensure_size(vector_container<Vector, Device>& vec,std::size_t size){
		vec().resize(size);
	}
}

///\brief Ensures that the matrix has the right size.
///
///Tries to resize mat. If the matrix expression can't be resized a debug assertion is thrown.
template<class Matrix, class Device>
void ensure_size(matrix_expression<Matrix, Device>& mat,std::size_t rows, std::size_t columns){
	detail::ensure_size(mat(),rows,columns);
}
///\brief Ensures that the vector has the right size.
///
///Tries to resize vec. If the vector expression can't be resized a debug assertion is thrown.
template<class Vector, class Device>
void ensure_size(vector_expression<Vector, Device>& vec,std::size_t size){
	detail::ensure_size(vec(),size);
}


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

template<class E1, class E2>
struct common_value_type
: public std::common_type<
	typename E1::value_type,
	typename E2::value_type
>{};
}

#endif
