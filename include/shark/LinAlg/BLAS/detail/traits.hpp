//===========================================================================
/*!
 * 
 *
 * \brief       -
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

#ifndef SHARK_LINALG_BLAS_DETAIL_TRAITS_HPP
#define SHARK_LINALG_BLAS_DETAIL_TRAITS_HPP

#include <complex>

#include "../fwd.hpp"
#include "iterator.hpp"
#include "returntype_deduction.hpp"
#include "../expression_types.hpp"

#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/is_float.hpp>
#include <boost/type_traits/is_integral.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/and.hpp>

#include <boost/range/iterator.hpp>

namespace shark {
namespace blas {
	
template<class T>
struct real_traits{
	typedef T type;
};

template<class T>
struct real_traits<std::complex<T> >{
	typedef T type;
};

// Use Joel de Guzman's return type deduction
// uBLAS assumes a common return type for all binary arithmetic operators
template<class X, class Y>
struct promote_traits {
	typedef type_deduction_detail::base_result_of<X, Y> base_type;
	static typename base_type::x_type x;
	static typename base_type::y_type y;
	static const std::size_t size = sizeof(
	        type_deduction_detail::test<
	        typename base_type::x_type
	        , typename base_type::y_type
	        >(x + y)     // Use x+y to stand of all the arithmetic actions
	        );

	static const std::size_t index = (size / sizeof(char)) - 1;
	typedef typename boost::mpl::at_c<
	typename base_type::types, index>::type id;
	typedef typename id::type promote_type;
};
// special case for bools. b1+b2 creates a boolean return type - which does not make sense
// for example when summing bools! therefore we use a signed int type
template<>
struct promote_traits<bool, bool> {
	typedef int promote_type;
};

struct upper;
struct unit_upper;
	
///\brief Flag indicating that the matrix is lower triangular
struct lower{
	static const bool is_upper = false;
	static const bool is_unit = false;
	typedef upper transposed_orientation;
	
};
///\brief Flag indicating that the matrix is lower triangular and diagonal elements are to be assumed as 1
struct unit_lower{
	static const bool is_upper = false;
	static const bool is_unit = true;
	typedef unit_upper transposed_orientation;
};
	
///\brief Flag indicating that the matrix is upper triangular
struct upper{
	static const bool is_upper = true;
	static const bool is_unit = false;
	typedef lower transposed_orientation;
};
///\brief Flag indicating that the matrix is upper triangular and diagonal elements are to be assumed as 1
struct unit_upper{
	static const bool is_upper = true;
	static const bool is_unit = true;
	typedef unit_lower transposed_orientation;
};


//structure types
struct linear_structure{};
struct packed_structure{};

// forward declaration
struct column_major;

// This traits class defines storage layout and it's properties
// matrix (i,j) -> storage [i * size_i + j]
struct row_major:public linear_structure{
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef row_major orientation;
	typedef column_major transposed_orientation;
	template<class T>
	struct sparse_element{
		size_type i;
		size_type j;
		T value;
		
		bool operator<(sparse_element const& other)const{
			if(i == other.i)
				return j< other.j;
			else
				return i < other.i;
		}
		
	};

	// Indexing conversion to storage element
	static size_type element(size_type i, size_type size_i, size_type j, size_type size_j) {
		SIZE_CHECK(i < size_i);
		SIZE_CHECK(j < size_j);
		return i * size_j + j;
	}
	static size_type address(size_type i, size_type size_i, size_type j, size_type size_j) {
		SIZE_CHECK(i < size_i);
		SIZE_CHECK(j < size_j);
		return i * size_j + j;
	}

	// Major and minor indices
	static size_type index_M(size_type index1, size_type /* index2 */) {
		return index1;
	}
	static size_type index_m(size_type /* index1 */, size_type index2) {
		return index2;
	}
	static size_type size_M(size_type size_i, size_type /* size_j */) {
		return size_i;
	}
	static size_type size_m(size_type /* size_i */, size_type size_j) {
		return size_j;
	}
	
	static size_type stride1(size_type /*size_i*/, size_type size_j){
		return size_j;
	}
	static size_type stride2(size_type /*size_i*/, size_type /*size_j*/){
		return 1;
	}
	
	static size_type  triangular_index(size_type i, size_type j, size_type size,lower){
		return i*(i+1)/2+j; 
	}
	
	static size_type  triangular_index(size_type i, size_type j, size_type size,upper){
		return (i*(2*size-i+1))/2+j-i; 
	}
};

// This traits class defines storage layout and it's properties
// matrix (i,j) -> storage [i + j * size_i]
struct column_major:public linear_structure{
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef row_major transposed_orientation;
	template<class T>
	struct sparse_element{
		size_type i;
		size_type j;
		T value;
		
		bool operator<(sparse_element const& other)const{
			if(j == other.j)
				return i< other.i;
			else
				return j < other.j;
		}
		
	};

	// Indexing conversion to storage element
	static size_type element(size_type i, size_type size_i, size_type j, size_type size_j) {
		SIZE_CHECK(i < size_i);
		SIZE_CHECK(j < size_j);
		return i + j * size_i;
	}
	static size_type address(size_type i, size_type size_i, size_type j, size_type size_j) {
		SIZE_CHECK(i < size_i);
		SIZE_CHECK(j < size_j);
		return i + j * size_i;
	}

	// Major and minor indices
	static size_type index_M(size_type /* index1 */, size_type index2) {
		return index2;
	}
	static size_type index_m(size_type index1, size_type /* index2 */) {
		return index1;
	}
	static size_type size_M(size_type /* size_i */, size_type size_j) {
		return size_j;
	}
	static size_type size_m(size_type size_i, size_type /* size_j */) {
		return size_i;
	}
	
	static size_type stride1(size_type /*size_i*/, size_type /*size_j*/){
		return 1;
	}
	static size_type stride2(size_type size_i, size_type /*size_j*/){
		return size_i;
	}
	
	static size_type  triangular_index(size_type i, size_type j, size_type size,lower){
		return transposed_orientation::triangular_index(j,i,size,upper()); 
	}
	
	static size_type  triangular_index(size_type i, size_type j, size_type size,upper){
		return transposed_orientation::triangular_index(j,i,size,lower()); 
	}
};
struct unknown_orientation:public linear_structure
{typedef unknown_orientation transposed_orientation;};

//storage schemes for packed matrices
template<class Orientation, class TriangularType>
struct packed:public packed_structure{
	typedef  TriangularType triangular_type;
	typedef Orientation orientation;
	typedef packed<
		typename Orientation::transposed_orientation,
		typename TriangularType::transposed_orientation
	> transposed_orientation;
	
	typedef typename Orientation::size_type size_type;
	static bool non_zero(size_type i, size_type  j){
		return TriangularType::is_upper? j >= i: i >= j;
	}
	
	static size_type element(size_type i, size_type j, size_type size) {
		SIZE_CHECK(i <= size);
		SIZE_CHECK(j <= size);
		//~ SIZE_CHECK( non_zero(i,j));//lets end iterators fail!
		
		return orientation::triangular_index(i,j,size,TriangularType());
	}
	
	static size_type stride1(size_type size_i, size_type size_j){
		return orientation::stride1(size_i,size_j);
	}
	static size_type stride2(size_type size_i, size_type size_j){
		return orientation::stride2(size_i,size_j);
	}
};

// Storage tags -- hierarchical definition of storage characteristics
// this gives the real storage layout of the matix in memory
// packed_tag ->BLAS packed format and supports packed interface
// dense_tag -> dense storage scheme an dense interface supported
// sparse_tag -> sparse storage scheme and supports sparse interface.
// unknown_storage_tag -> no known storage scheme, only supports basic interface
struct unknown_storage_tag {};
struct sparse_tag:public unknown_storage_tag{};
struct dense_tag: public unknown_storage_tag{};
struct packed_tag: public unknown_storage_tag{};


template<class S1, class S2>
struct storage_restrict_traits {
	typedef S1 storage_category;
};
template<>
struct storage_restrict_traits<dense_tag, sparse_tag> {
	typedef sparse_tag storage_category;
};

template<class E>
struct closure: public boost::mpl::if_<
	boost::is_const<E>,
	typename E::const_closure_type,
	typename E::closure_type
>{};

template<class E>
struct reference: public boost::mpl::if_<
	boost::is_const<E>,
	typename E::const_reference,
	typename E::reference
>{};

template<class E>
struct pointer: public boost::mpl::if_<
	boost::is_const<E>,
	typename E::const_pointer,
	typename E::pointer
>{};
template<class E>
struct index_pointer: public boost::mpl::if_<
	boost::is_const<E>,
	typename E::const_index_pointer,
	typename E::index_pointer
>{};
	
template<class M>
struct row_iterator: public boost::mpl::if_<
	boost::is_const<M>,
	typename M::const_row_iterator,
	typename M::row_iterator
>{};
	
template<class M>
struct column_iterator: public boost::mpl::if_<
	boost::is_const<M>,
	typename M::const_column_iterator,
	typename M::column_iterator
>{};

template<class Matrix> 
struct major_iterator:public boost::mpl::if_<
	boost::is_same<typename Matrix::orientation, column_major>,
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
	typename column_iterator<M>::type major_end(M& m,std::size_t i, column_major){
		return m.column_end(i);
	}
	template<class M>
	typename row_iterator<M>::type major_end(M& m,std::size_t i, row_major){
		return m.row_end(i);
	}
}

template<class M>
typename major_iterator<M const>::type major_begin(matrix_expression<M> const& m, std::size_t i){
	return detail::major_begin(m(),i, typename M::orientation());
}
template<class M>
typename major_iterator<M const>::type major_end(matrix_expression<M> const& m, std::size_t i){
	return detail::major_end(m(),i, typename M::orientation());
}
template<class M>
typename major_iterator<M>::type major_begin(matrix_expression<M>& m, std::size_t i){
	return detail::major_begin(m(),i, typename M::orientation());
}
template<class M>
typename major_iterator<M>::type major_end(matrix_expression<M>& m, std::size_t i){
	return detail::major_end(m(),i, typename M::orientation());
}

///\brief Determines a good vector type storing an expression returning values of type T and having a certain iterator category.
template<class ValueType, class IteratorTag>
struct vector_temporary_type;
///\brief Determines a good vector type storing an expression returning values of type T and having a certain iterator category.
template<class ValueType, class Orientation, class IteratorTag>
struct matrix_temporary_type;

/// For the creation of temporary vectors in the assignment of proxies
template <class E>
struct vector_temporary{
	typedef typename vector_temporary_type<
		typename E::value_type,
		typename boost::mpl::eval_if<
			typename boost::is_base_of<vector_expression<E>,E>::type,
			boost::range_iterator<E>,
			major_iterator<E>
		>::type::iterator_category
	>::type type;
};

/// For the creation of temporary matrix in the assignment of proxies
template <class E>
struct matrix_temporary{
	typedef typename matrix_temporary_type<
		typename E::value_type,
		typename E::orientation,
		typename boost::mpl::eval_if<
			typename boost::is_base_of<vector_expression<E>,E>::type,
			boost::range_iterator<E>,
			major_iterator<E>
		>::type::iterator_category
	>::type type;
};

/// For the creation of transposed temporary matrix in the assignment of proxies
template <class E>
struct transposed_matrix_temporary{
	typedef typename matrix_temporary_type<
		typename E::value_type,
		typename E::orientation::transposed_orientation,
		typename boost::mpl::eval_if<
			typename boost::is_base_of<vector_expression<E>,E>::type,
			boost::range_iterator<E>,
			major_iterator<E>
		>::type::iterator_category
	>::type type;
};

namespace detail{
	template<class Matrix>
	void ensure_size(matrix_expression<Matrix>& mat,std::size_t rows, std::size_t columns){
		SIZE_CHECK(mat().size1() == rows);
		SIZE_CHECK(mat().size2() == columns);
	}
	template<class Matrix>
	void ensure_size(matrix_container<Matrix>& mat,std::size_t rows, std::size_t columns){
		mat().resize(rows,columns);
	}
	template<class Vector>
	void ensure_size(vector_expression<Vector>& vec,std::size_t size){
		SIZE_CHECK(vec().size() == size);
	}
	template<class Vector>
	void ensure_size(vector_container<Vector>& vec,std::size_t size){
		vec().resize(size);
	}
}

///\brief Ensures that the matrix has the right size.
///
///Tries to resize mat. If the matrix expression can't be resized a debug assertion is thrown.
template<class Matrix>
void ensure_size(matrix_expression<Matrix>& mat,std::size_t rows, std::size_t columns){
	detail::ensure_size(mat(),rows,columns);
}
///\brief Ensures that the vector has the right size.
///
///Tries to resize vec. If the vector expression can't be resized a debug assertion is thrown.
template<class Vector>
void ensure_size(vector_expression<Vector>& vec,std::size_t size){
	detail::ensure_size(vec(),size);
}

}
}

#endif
