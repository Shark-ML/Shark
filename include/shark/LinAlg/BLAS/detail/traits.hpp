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
#include "functional.hpp"
#include "../expression_types.hpp"

#include <boost/mpl/eval_if.hpp>

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
struct closure: public boost::mpl::if_<
	std::is_const<E>,
	typename E::const_closure_type,
	typename E::closure_type
>{};
	
template<class E>
struct const_expression : public boost::mpl::if_c<
	std::is_base_of<vector_container<typename std::remove_const<E>::type,typename E::device_type>, E >::value
	||std::is_base_of<matrix_container<typename std::remove_const<E>::type,typename E::device_type>, E >::value,
	E const,
	typename E::const_closure_type
>{};

template<class E>
struct reference: public boost::mpl::if_<
	std::is_const<E>,
	typename E::const_reference,
	typename E::reference
>{};

template<class E>
struct storage: public boost::mpl::if_<
	std::is_const<E>,
	typename E::const_storage_type,
	typename E::storage_type
>{};
	
template<class M>
struct row_iterator: public boost::mpl::if_<
	std::is_const<M>,
	typename M::const_row_iterator,
	typename M::row_iterator
>{};
	
template<class M>
struct column_iterator: public boost::mpl::if_<
	std::is_const<M>,
	typename M::const_column_iterator,
	typename M::column_iterator
>{};

template<class Matrix> 
struct major_iterator:public boost::mpl::if_<
	std::is_same<typename Matrix::orientation, column_major>,
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
	using transform_iterator = iterators::transform_iterator<Iterator, Functor>;

	template <class Iterator>
	using subrange_iterator = iterators::subrange_iterator<Iterator>;
	
	template<class Iterator1, class Iterator2, class Functor>
	using binary_transform_iterator = iterators::binary_transform_iterator<Iterator1,Iterator2, Functor>;
	
	template<class T>
	using constant_iterator = iterators::constant_iterator<T>;
	
	template<class T>
	using one_hot_iterator = iterators::one_hot_iterator<T>;
	
	template<class Closure>
	using indexed_iterator = iterators::indexed_iterator<Closure>;
	
	//functors
	template<class T>
	using add = functors::scalar_binary_plus<T>;
	template<class T>
	using subtract = functors::scalar_binary_minus<T>;
	template<class T>
	using multiply = functors::scalar_binary_multiply<T>;
	template<class T>
	using divide = functors::scalar_binary_divide<T>;
	template<class T>
	using multiply_and_add = functors::scalar_binary_multiply_and_add<T>;
	template<class T>
	using multiply_assign = functors::scalar_binary_multiply_assign<T>;
	template<class T>
	using pow = functors::scalar_binary_pow<T>;
	template<class T>
	using multiply_scalar = functors::scalar_multiply1<T>;
	template<class T>
	using safe_divide = functors::scalar_binary_safe_divide<T>;
	
	//math unary functions
	template<class T>
	using log = functors::scalar_log<T>;
	template<class T>
	using exp = functors::scalar_exp<T>;
	template<class T>
	using tanh = functors::scalar_tanh<T>;
	template<class T>
	using sqrt = functors::scalar_sqrt<T>;
	template<class T>
	using abs = functors::scalar_abs<T>;
	template<class T>
	using sqr = functors::scalar_sqr<T>;
	template<class T>
	using soft_plus = functors::scalar_soft_plus<T>;
	template<class T>
	using sigmoid = functors::scalar_sigmoid<T>;
	template<class T>
	using inv = functors::scalar_inverse<T>;
	
	//min/max
	template<class T>
	using min = functors::scalar_binary_min<T>;
	template<class T>
	using max = functors::scalar_binary_max<T>;
	
	//comparison
	template<class T>
	using less = functors::scalar_less_than<T>;
	template<class T>
	using less_equal = functors::scalar_less_equal_than<T>;
	template<class T>
	using bigger = functors::scalar_bigger_than<T>;
	template<class T>
	using bigger_equal = functors::scalar_bigger_equal_than<T>;
	template<class T>
	using equal = functors::scalar_equal<T>;
	template<class T>
	using not_equal = functors::scalar_not_equal<T>;
};

template<class E1, class E2>
struct common_value_type
: public std::common_type<
	typename E1::value_type,
	typename E2::value_type
>{};
}

#endif
