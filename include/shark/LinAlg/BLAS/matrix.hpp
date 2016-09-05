/*!
 * \brief       Dense Matrix class
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
#ifndef SHARK_LINALG_BLAS_MATRIX_HPP
#define SHARK_LINALG_BLAS_MATRIX_HPP

#include "assignment.hpp"
#include <array>
#include <initializer_list>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>

namespace shark {
namespace blas {

/** \brief A dense matrix of values of type \c T.
 *
 * For a \f$(m \times n)\f$-dimensional matrix and \f$ 0 \leq i < m, 0 \leq j < n\f$, every element \f$ m_{i,j} \f$ is mapped to
 * the \f$(i.n + j)\f$-th element of the container for row major orientation or the \f$ (i + j.m) \f$-th element of
 * the container for column major orientation. In a dense matrix all elements are represented in memory in a
 * contiguous chunk of memory by definition.
 *
 * Orientation can also be specified, otherwise a \c row_major is used.
 *
 * \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
 * \tparam L the storage organization. It can be either \c row_major or \c column_major. Default is \c row_major
 */
template<class T, class L=row_major>
class matrix:public matrix_container<matrix<T, L> > {
	typedef matrix<T, L> self_type;
	typedef std::vector<T> array_type;
public:
	typedef typename array_type::size_type size_type;
	typedef typename array_type::difference_type difference_type;
	typedef typename array_type::value_type value_type;
	typedef value_type scalar_type;
	typedef typename array_type::const_reference const_reference;
	typedef typename array_type::reference reference;
	typedef const T* const_pointer;
	typedef T* pointer;

	typedef std::size_t index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type index_pointer;

	typedef matrix_reference<self_type const> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef dense_tag storage_category;
	typedef elementwise_tag evaluation_category;
	typedef L orientation;

	// Construction

	/// \brief Default dense matrix constructor. Make a dense matrix of size (0,0)
	matrix():m_size1(0), m_size2(0){}
	
	/// \brief Constructor from a nested initializer list.
	///
	/// Constructs a matrix like this: m = {{1,2},{3,4}}.
	/// \param list The nested initializer list storing the values of the matrix.
	matrix(std::initializer_list<std::initializer_list<T> > list)
	: m_size1(list.size())
	, m_size2(list.begin()->size())
	, m_data(m_size1*m_size2){
		auto pos = list.begin();
		for(std::size_t i = 0; i != list.size(); ++i,++pos){
			SIZE_CHECK(pos->size() == m_size2);
			std::copy(pos->begin(),pos->end(),row_begin(i));
		}
	}

	/// \brief Dense matrix constructor with defined size
	/// \param size1 number of rows
	/// \param size2 number of columns
	matrix(size_type size1, size_type size2)
	:m_size1(size1)
	, m_size2(size2)
	, m_data(size1 * size2) {}

	/// \brief  Dense matrix constructor with defined size a initial value for all the matrix elements
	/// \param size1 number of rows
	/// \param size2 number of columns
	/// \param init initial value assigned to all elements
	matrix(size_type size1, size_type size2, value_type const& init)
	: m_size1(size1)
	, m_size2(size2)
	, m_data(size1 * size2, init) {}

	/// \brief Copy-constructor of a dense matrix
	///\param m is a dense matrix
	matrix(matrix const& m) = default;
			
	/// \brief Move-constructor of a dense matrix
	///\param m is a dense matrix
	//~ matrix(matrix&& m) = default; //vc++ can not default this
	matrix(matrix&& m):m_size1(m.m_size1), m_size2(m.m_size2), m_data(std::move(m.m_data)){}

	/// \brief Constructor of a dense matrix from a matrix expression.
	/// 
	/// Constructs the matrix by evaluating the expression and assigning the
	/// results to the newly constructed matrix using a call to assign.
	///
	/// \param e is a matrix expression
	template<class E>
	matrix(matrix_expression<E> const& e)
	: m_size1(e().size1())
	, m_size2(e().size2())
	, m_data(m_size1 * m_size2) {
		assign(*this,e);
	}
	
	// Assignment
	
	/// \brief Assigns m to this
	matrix& operator = (matrix const& m) = default;
	
	/// \brief Move-Assigns m to this
	//~ matrix& operator = (matrix&& m) = default;//vc++ can not default this
	matrix& operator = (matrix&& m) {
		m_size1 = m.m_size1;
		m_size2 = m.m_size2;
		m_data = std::move(m.m_data);
		return *this;
	}

	
	/// \brief Assigns m to this
	/// 
	/// evaluates the expression and assign the
	/// results to this using a call to assign.
	/// As containers are assumed to not overlap, no temporary is created
	///
	/// \param m is a matrix expression
	template<class C>
	matrix& operator = (matrix_container<C> const& m) {
		resize(m().size1(), m().size2());
		assign(*this, m);
		return *this;
	}
	/// \brief Assigns e to this
	/// 
	/// evaluates the expression and assign the
	/// results to this using a call to assign.
	/// A temporary is created to prevent aliasing.
	///
	/// \param e is a matrix expression
	template<class E>
	matrix& operator = (matrix_expression<E> const& e) {
		self_type temporary(e);
		swap(temporary);
		return *this;
	}

	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the number of rows of the matrix.
	size_type size1() const {
		return m_size1;
	}
	///\brief Returns the number of columns of the matrix.
	size_type size2() const {
		return m_size2;
	}
	
	///\brief Returns the stride in memory between two rows.
	difference_type stride1()const{
		return orientation::stride1(m_size1,m_size2);
	}
	///\brief Returns the stride in memory between two columns.
	difference_type stride2()const{
		return orientation::stride2(m_size1,m_size2);
	}
	
	///\brief Returns the pointer to the beginning of the matrix storage
	///
	/// Grants low-level access to the matrix internals. Element order depends on whether the matrix is row_major or column_major.
	/// to access element _(i,j) use storage()[i*stride1()+j*stride2()].
	const_pointer storage()const{
		return m_data.data();
	}
	
	///\brief Returns the pointer to the beginning of the matrix storage
	///
	/// Grants low-level access to the matrix internals. Element order depends on whether the matrix is row_major or column_major.
	/// to access element (i,j) use storage()[i*stride1()+j*stride2()].
	pointer storage(){
		return m_data.data();
	}
	
	// ---------
	// High level interface
	// ---------

	// Resizing
	/// \brief Resize a matrix to new dimensions. If resizing is performed, the data is not preserved.
	/// \param size1 the new number of rows
	/// \param size2 the new number of colums
	void resize(size_type size1, size_type size2) {
		m_data.resize(size1* size2);
		m_size1 = size1;
		m_size2 = size2;
	}
	
	void clear(){
		std::fill(m_data.begin(), m_data.end(), value_type/*zero*/());
	}

	// Element access
	const_reference operator()(index_type i, index_type j) const {
		SIZE_CHECK(i < size1());
		SIZE_CHECK(j < size2());
		return m_data[orientation::element(i, m_size1, j, m_size2)];
	}
	reference operator()(index_type i, index_type j) {
		SIZE_CHECK(i < size1());
		SIZE_CHECK(j < size2());
		return m_data[orientation::element(i, m_size1, j, m_size2)];
	}
	
	void set_element(size_type i, size_type j,value_type t){
		SIZE_CHECK(i < size1());
		SIZE_CHECK(j < size2());
		m_data[orientation::element(i, m_size1, j, m_size2)]  = t;
	}

	// Swapping
	void swap(matrix& m) {
		std::swap(m_size1, m.m_size1);
		std::swap(m_size2, m.m_size2);
		m_data.swap(m.m_data);
	}
	friend void swap(matrix& m1, matrix& m2) {
		m1.swap(m2);
	}
	
	friend void swap_rows(matrix& a, index_type i, matrix& b, index_type j){
		SIZE_CHECK(i < a.size1());
		SIZE_CHECK(j < b.size1());
		SIZE_CHECK(a.size2() == b.size2());
		for(std::size_t k = 0; k != a.size2(); ++k){
			std::swap(a(i,k),b(j,k));
		}
	}
	
	friend void swap_rows(matrix& a, index_type i, index_type j) {
		if(i == j) return;
		swap_rows(a,i,a,j);
	}
	
	
	friend void swap_columns(matrix& a, index_type i, matrix& b, index_type j){
		SIZE_CHECK(i < a.size2());
		SIZE_CHECK(j < b.size2());
		SIZE_CHECK(a.size1() == b.size1());
		for(std::size_t k = 0; k != a.size1(); ++k){
			std::swap(a(k,i),b(k,j));
		}
	}
	
	friend void swap_columns(matrix& a, index_type i, index_type j) {
		if(i == j) return;
		swap_columns(a,i,a,j);
	}

	//Iterators
	typedef dense_storage_iterator<value_type> row_iterator;
	typedef dense_storage_iterator<value_type> column_iterator;
	typedef dense_storage_iterator<value_type const> const_row_iterator;
	typedef dense_storage_iterator<value_type const> const_column_iterator;

	const_row_iterator row_begin(index_type i) const {
		return const_row_iterator(storage() + i*stride1(),0,stride2());
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(storage() + i*stride1()+stride2()*size2(),size2(),stride2());
	}
	row_iterator row_begin(index_type i){
		return row_iterator(storage() + i*stride1(),0,stride2());
	}
	row_iterator row_end(index_type i){
		return row_iterator(storage() + i*stride1()+stride2()*size2(),size2(),stride2());
	}
	
	const_row_iterator column_begin(std::size_t j) const {
		return const_column_iterator(storage() + j*stride2(),0,stride1());
	}
	const_column_iterator column_end(std::size_t j) const {
		return const_column_iterator(storage() + j*stride2()+ stride1()*size1(),size1(),stride1());
	}
	column_iterator column_begin(std::size_t j){
		return column_iterator(storage() + j*stride2(),0,stride1());
	}
	column_iterator column_end(std::size_t j){
		return column_iterator(storage() + j * stride2()+ stride1() * size1(), size1(), stride1());
	}
	
	typedef typename blas::major_iterator<self_type>::type major_iterator;
	
	//sparse interface
	major_iterator set_element(major_iterator pos, index_type index, value_type value) {
		RANGE_CHECK(pos.index() == index);
		*pos=value;
		return pos;
	}
	
	major_iterator clear_element(major_iterator elem) {
		*elem = value_type();
		return elem+1;
	}
	
	major_iterator clear_range(major_iterator start, major_iterator end) {
		std::fill(start,end,value_type());
		return end;
	}
	
	void reserve(size_type non_zeros) {}
	void reserve_row(std::size_t, std::size_t){}
	void reserve_column(std::size_t, std::size_t){}

	// Serialization
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */) {

		// we need to copy to a collection_size_type to get a portable
		// and efficient boost::serialization
		boost::serialization::collection_size_type s1(m_size1);
		boost::serialization::collection_size_type s2(m_size2);

		// serialize the sizes
		ar& boost::serialization::make_nvp("size1",s1)
		& boost::serialization::make_nvp("size2",s2);

		// copy the values back if loading
		if (Archive::is_loading::value) {
			m_size1 = s1;
			m_size2 = s2;
		}
		ar& boost::serialization::make_nvp("data",m_data);
	}

private:
	size_type m_size1;
	size_type m_size2;
	array_type m_data;
};
template<class T, class L>
struct matrix_temporary_type<T,L,dense_random_access_iterator_tag>{
	typedef matrix<T,L> type;
};

template<class T>
struct matrix_temporary_type<T,unknown_orientation,dense_random_access_iterator_tag>{
	typedef matrix<T,row_major> type;
};

template<class T, class Orientation>
struct const_expression<matrix<T,Orientation> >{
	typedef matrix<T,Orientation> const type;
};
template<class T, class Orientation>
struct const_expression<matrix<T,Orientation> const>{
	typedef matrix<T,Orientation> const type;
};

}
}

#endif
