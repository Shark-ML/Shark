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
struct major_iterator: public std::conditional<
	std::is_const<M>::value,
	typename M::const_major_iterator,
	typename M::major_iterator
>{};
	
template<class M>
struct minor_iterator: public std::conditional<
	std::is_const<M>::value,
	typename M::const_minor_iterator,
	typename M::minor_iterator
>{};

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
		REMORA_SIZE_CHECK(mat().size1() == rows);
		REMORA_SIZE_CHECK(mat().size2() == columns);
	}
	template<class Matrix, class Device>
	void ensure_size(matrix_container<Matrix, Device>& mat,std::size_t rows, std::size_t columns){
		mat().resize(rows,columns);
	}
	template<class Vector, class Device>
	void ensure_size(vector_expression<Vector, Device>& vec,std::size_t size){
		REMORA_SIZE_CHECK(vec().size() == size);
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

template<class E1, class E2>
struct common_value_type
: public std::common_type<
	typename E1::value_type,
	typename E2::value_type
>{};

template<class E>
struct ExpressionToFunctor;

template<class E, class Device>
auto to_functor(matrix_expression<E, Device> const& e) -> decltype(ExpressionToFunctor<E>::transform(e())){
	return ExpressionToFunctor<E>::transform(e());
}

template<class E, class Device>
auto to_functor(vector_expression<E, Device> const& e) -> decltype(ExpressionToFunctor<E>::transform(e())){
	return ExpressionToFunctor<E>::transform(e());
}

template<class M, class Device>
typename M::size_type major_size(matrix_expression<M, Device> const& m){
	return M::orientation::index_M(m().size1(),m().size2());
}
template<class M, class Device>
typename M::size_type minor_size(matrix_expression<M, Device> const& m){
	return M::orientation::index_m(m().size1(),m().size2());
}

}

#include "../cpu/traits.hpp"
#ifdef REMORA_USE_GPU
#include "../gpu/traits.hpp"
#endif

#endif


