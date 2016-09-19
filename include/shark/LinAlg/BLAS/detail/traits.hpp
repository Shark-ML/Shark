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

#ifndef SHARK_LINALG_BLAS_DETAIL_TRAITS_HPP
#define SHARK_LINALG_BLAS_DETAIL_TRAITS_HPP

#include "iterator.hpp"
#include "evaluation_tags.hpp"
#include "../expression_types.hpp"

#include <boost/mpl/eval_if.hpp>
#include <boost/range/iterator.hpp>

#include <complex>
#include <type_traits>

namespace shark {
namespace blas {
	
// forward declaration
struct row_major;
struct column_major;

struct unknown_storage{
	typedef unknown_tag storage_tag;
	typedef unknown_storage row_storage;
};

template<class T>
struct dense_vector_storage{
	typedef dense_tag storage_tag;
	T* values;
	std::size_t stride;
	
	dense_vector_storage<T> sub_region(std::size_t offset){
		return {values+offset*stride, stride};
	}
};

template<class T, class I>
struct sparse_vector_storage{
	typedef sparse_tag storage_tag;
	T* values;
	I* indices;
	std::size_t nnz;
};

template<class T>
struct dense_matrix_storage{
	typedef dense_tag storage_tag;
	typedef dense_vector_storage<T> row_storage;
	T* values;
	std::size_t leading_dimension;
	
	template<class Orientation>
	dense_matrix_storage<T> sub_region(std::size_t offset1, std::size_t offset2, Orientation){
		std::size_t offset_major = Orientation::index_M(offset1,offset2);
		std::size_t offset_minor = Orientation::index_m(offset1,offset2);
		return {values+offset_major*leading_dimension+offset_minor, leading_dimension};
	}
	
	template<class Orientation>
	row_storage row(std::size_t i, Orientation){
		return {values + i * Orientation::index_M(leading_dimension,1), Orientation::index_m(leading_dimension,1)};
	}
	template<class Orientation>
	row_storage diag(){
		return {values, leading_dimension+1};
	}
};

template<class T>
struct packed_matrix_storage{
	typedef packed_tag storage_tag;
	typedef dense_vector_storage<T> row_storage;
	T* values;
	std::size_t nnz;
};

template<class T, class I>
struct sparse_matrix_storage{
	typedef sparse_tag storage_tag;
	typedef sparse_vector_storage<T,I> row_storage;
	T* values;
	I* indices;
	I* outer_indices_begin;
	I* outer_indices_end;
	
	template<class Orientation>
	row_storage row(std::size_t i, Orientation){
		static_assert(std::is_same<Orientation,row_major>::value, "sparse matrix has wrong orientation for row/column");
		return {values + outer_indices_begin[i], indices + outer_indices_begin[i],outer_indices_end[i] - outer_indices_begin[i]};
	}
};
	
template<class T>
struct real_traits{
	typedef T type;
};

template<class T>
struct real_traits<std::complex<T> >{
	typedef T type;
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
struct triangular_structure{};

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
};

// This traits class defines storage layout and it's properties
// matrix (i,j) -> storage [i + j * size_i]
struct column_major:public linear_structure{
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef column_major orientation;
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
};
struct unknown_orientation:public linear_structure{
	typedef unknown_orientation orientation;
	typedef unknown_orientation transposed_orientation;
};

//storage schemes for packed matrices
template<class Orientation, class TriangularType>
struct triangular: public triangular_structure{
public:
	static const bool is_upper = TriangularType::is_upper;
	static const bool is_unit = TriangularType::is_unit;
	typedef TriangularType triangular_type;
	typedef Orientation orientation;
	typedef triangular<
		typename Orientation::transposed_orientation,
		typename TriangularType::transposed_orientation
	> transposed_orientation;
	
	typedef typename Orientation::size_type size_type;
	static bool non_zero(size_type i, size_type  j){
		return TriangularType::is_upper? j >= i: i >= j;
	}
	
	template<class StorageTag>
	static size_type element(size_type i, size_type j, size_type size, StorageTag tag) {
		SIZE_CHECK(i <= size);
		SIZE_CHECK(j <= size);
		//~ SIZE_CHECK( non_zero(i,j));//lets end iterators fail!
		return triangular_index(i,j,size,TriangularType(), Orientation(), tag);
	}
private:
	static size_type  triangular_index(size_type i, size_type j, size_type size,lower, row_major, packed_tag){
		return i*(i+1)/2+j; 
	}
	static size_type  triangular_index(size_type i, size_type j, size_type size,upper, row_major, packed_tag){
		return (i*(2*size-i+1))/2+j-i; 
	}
	static size_type  triangular_index(size_type i, size_type j, size_type size,lower, row_major, dense_tag){
		return row_major::element(i,size,j,size); 
	}
	static size_type  triangular_index(size_type i, size_type j, size_type size,upper, row_major, dense_tag){
		return column_major::element(i,size,j,size); 
	}
	template<class TriangT, class StructT>
	static size_type  triangular_index(size_type i, size_type j, size_type size,TriangT, column_major, StructT s){
		return triangular_index(j,i,size,typename TriangT::transposed_orientation(),row_major(), s);
	}
};

template<class E>
struct closure: public boost::mpl::if_<
	std::is_const<E>,
	typename E::const_closure_type,
	typename E::closure_type
>{};
	
template<class E>
struct const_expression : public boost::mpl::if_c<
	std::is_base_of<blas::vector_container<typename std::remove_const<E>::type,typename E::device_type>, E >::value
	||std::is_base_of<blas::matrix_container<typename std::remove_const<E>::type,typename E::device_type>, E >::value,
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
	typename column_iterator<M>::type major_end(M& m,std::size_t i, column_major){
		return m.column_end(i);
	}
	template<class M>
	typename row_iterator<M>::type major_end(M& m,std::size_t i, row_major){
		return m.row_end(i);
	}
}

template<class M, class Device>
typename major_iterator<M const>::type major_begin(matrix_expression<M, Device> const& m, std::size_t i){
	return detail::major_begin(m(),i, typename M::orientation());
}
template<class M, class Device>
typename major_iterator<M const>::type major_end(matrix_expression<M, Device> const& m, std::size_t i){
	return detail::major_end(m(),i, typename M::orientation());
}
template<class M, class Device>
typename major_iterator<M>::type major_begin(matrix_expression<M, Device>& m, std::size_t i){
	return detail::major_begin(m(),i, typename M::orientation());
}
template<class M, class Device>
typename major_iterator<M>::type major_end(matrix_expression<M, Device>& m, std::size_t i){
	return detail::major_end(m(),i, typename M::orientation());
}

///\brief Determines a good vector type storing an expression returning values of type T having a certain evaluation category on a specific device.
template<class ValueType, class Cateogry, class Device>
struct vector_temporary_type;
///\brief Determines a good vector type storing an expression returning values of type T having a certain evaluation category on a specific device.
template<class ValueType, class Orientation, class Cateogry, class Device>
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

}
}

#endif
