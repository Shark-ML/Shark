/*!
 * 
 *
 * \brief       Matrix proxy classes.
 * 
 * 
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

#ifndef SHARK_LINALG_BLAS_MATRIX_PROXY_HPP
#define SHARK_LINALG_BLAS_MATRIX_PROXY_HPP

#include "assignment.hpp"

namespace shark {
namespace blas {
	
///\brief Wraps another expression as a reference.
template<class M>
class matrix_reference:public matrix_expression<matrix_reference<M> > {
public:
	typedef typename M::size_type size_type;
	typedef typename M::difference_type difference_type;
	typedef typename M::value_type value_type;
	typedef typename M::scalar_type scalar_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::const_pointer const_pointer;
	typedef typename pointer<M>::type pointer;

	typedef typename M::index_type index_type;
	typedef typename M::const_index_pointer const_index_pointer;
	typedef typename index_pointer<M>::type index_pointer;

	typedef matrix_reference<M const> const_closure_type;
	typedef matrix_reference<M> closure_type;
	typedef typename M::orientation orientation;
	typedef typename M::storage_category storage_category;
	typedef elementwise_tag evaluation_category;


	// Construction and destruction
	matrix_reference(M& m):m_expression(&m) {}
	template<class E>
	matrix_reference(matrix_reference<E> const& other)
		:m_expression(&other.expression()){}
		
	// Accessors

	M& expression() const {
		return *m_expression;
	}
	M& expression() {
		return *m_expression;
	}
	
	///\brief Returns the number of rows of the matrix.
	size_type size1() const {
		return expression().size1();
	}
	///\brief Returns the number of columns of the matrix.
	size_type size2() const {
		return expression().size2();
	}
	
	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the stride in memory between two rows.
	difference_type stride1()const{
		return expression().stride1();
	}
	///\brief Returns the stride in memory between two columns.
	difference_type stride2()const{
		return expression().stride2();
	}
	
	///\brief Returns the pointer to the beginning of the matrix storage.
	///
	/// Grants low-level access to the matrix internals. Element order depends on whether the matrix is row_major or column_major.
	/// to access element (i,j) use storage()[i*stride1()+j*stride2()].
	pointer storage()const{
		return expression().storage();
	}
	
	// ---------
	// Sparse low level interface
	// ---------
	
	/// \brief Number of nonzero elements of the matrix.
	size_type nnz()const{
		return expression().nnz();
	}
	/// \brief Array of values of the nonzero elements.
	const_pointer values()const{
		return expression().values();
	}
	
	/// \brief Array of indices of the nonzero elements.
	///
	/// Note that there is a pair of indices needed:
	/// When accessing the j-th element in row i you have to write code like this:
	/// index_type start = outer_indices()[i] //aquire start of the i-th row
	/// index = inner_indices()[start+j];
	/// All elements in the row are contained in the range [outer_indices()[i],outer_indices_end()[i])
	/// there might be gaps between the end of the one line and the start of the next!
	index_pointer inner_indices()const{
		return expression().inner_indices();
	}
	
	///\brief Returns an array containing the start of the rows.
	///
	/// See documentation of inner_indices() for more details
	index_pointer outer_indices()const{
		return expression().outer_indices();
	}
	
	///\brief Returns an array containing the end of the rows.
	///
	/// See documentation of inner_indices() for more details
	index_pointer outer_indices_end()const{
		return expression().outer_indices_end();
	}
	
	/// \brief Returns the number of nonzero elements in the i-th row/column.
	size_type inner_nnz(index_type i) const {
		return expression().inner_nnz(i);
	}

	// ---------
	// High level interface
	// ---------

	// Element access
	reference operator()(index_type i, index_type j) const {
		return expression()(i, j);
	}
	
	void set_element(size_type i, size_type j,value_type t){
		expression().set_element(i,j,t);
	}


	// Assignment
	template<class E>
	matrix_reference& operator = (matrix_expression<E> const& e) {
		expression() = e();
		return *this;
	}

	// Iterator types
	typedef typename row_iterator<M>::type row_iterator;
	typedef row_iterator const_row_iterator;
	typedef typename column_iterator<M>::type column_iterator;
	typedef column_iterator const_column_iterator;

	// Iterators are the iterators of the referenced expression.
	const_row_iterator row_begin(index_type i) const {
		return expression().row_begin(i);
	}
	const_row_iterator row_end(index_type i) const {
		return expression().row_end(i);
	}
	row_iterator row_begin(index_type i){
		return expression().row_begin(i);
	}
	row_iterator row_end(index_type i){
		return expression().row_end(i);
	}
	
	const_column_iterator column_begin(index_type j) const {
		return expression().column_begin(j);
	}
	const_column_iterator column_end(index_type j) const {
		return expression().column_end(j);
	}
	column_iterator column_begin(index_type j){
		return expression().column_begin(j);
	}
	column_iterator column_end(index_type j){
		return expression().column_end(j);
	}
	
	row_iterator set_element(row_iterator pos, index_type index, value_type value) {
		return expression().set_element(pos,index,value);
	}
	
	row_iterator clear_range(row_iterator start, row_iterator end) {
		return expression().clear_range(start,end);
	}
	
	row_iterator clear_element(row_iterator elem) {
		return expression().clear_element(elem);
	}
	
	void clear(){
		expression().clear();
	}
	
	void reserve(size_type non_zeros) {
		expression().reserve(non_zeros);
	}
	
	void reserve_row(size_type row, size_type non_zeros) {
		expression().reserve_row(row,non_zeros);
	}
	
	void reserve_column(size_type column, size_type non_zeros) {
		expression().reserve_column(column,non_zeros);
	}
	
	
	void swap(matrix_reference& m) {
		expression().swap(m.expression());
	}

	friend void swap(matrix_reference& m1, matrix_reference& m2) {
		m1.swap(m2);
	}
	
	void swap_rows(index_type i, index_type j){
		expression().swap_rows(i,j);
	}
	
	void swap_columns(index_type i, index_type j){
		expression().swap_columns(i,j);
	}
private:
	M* m_expression;
};

/// \brief Matrix transpose.
template<class M>
class matrix_transpose: public matrix_expression<matrix_transpose<M> > {
public:
	typedef typename M::size_type size_type;
	typedef typename M::difference_type difference_type;
	typedef typename M::value_type value_type;
	typedef typename M::scalar_type scalar_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::const_pointer const_pointer;
	typedef typename pointer<M>::type pointer;

	typedef typename M::index_type index_type;
	typedef typename M::const_index_pointer const_index_pointer;
	typedef typename index_pointer<M>::type index_pointer;

	typedef typename closure<M>::type matrix_closure_type;
	typedef matrix_transpose<typename const_expression<M>::type> const_closure_type;
	typedef matrix_transpose<M> closure_type;
	typedef typename M::orientation::transposed_orientation orientation;
	typedef typename M::storage_category storage_category;
	typedef elementwise_tag evaluation_category;

	// Construction and destruction
	explicit matrix_transpose(matrix_closure_type const& m):
		m_expression(m) {}
	
	//conversion closure->const_closure
	template<class E>
	matrix_transpose(
		matrix_transpose<E> const& m,
		typename boost::disable_if<
			boost::mpl::or_<
				boost::is_same<matrix_transpose<E>,matrix_transpose>,
				boost::is_same<matrix_transpose<E>,matrix_closure_type>
			> 
		>::type* dummy = 0
	):m_expression(m.expression()) {}

	// Expression accessors
	matrix_closure_type const& expression() const{
		return m_expression;
	}
	matrix_closure_type expression(){
		return m_expression;
	}
	
	///\brief Returns the number of rows of the matrix.
	size_type size1() const {
		return expression().size2();
	}
	///\brief Returns the number of columns of the matrix.
	size_type size2() const {
		return expression().size1();
	}
	
	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the stride in memory between two rows.
	difference_type stride1()const{
		return expression().stride2();
	}
	///\brief Returns the stride in memory between two columns.
	difference_type stride2()const{
		return expression().stride1();
	}
	
	// ---------
	// Sparse low level interface
	// ---------
	
	/// \brief Number of nonzero elements of the matrix.
	size_type nnz()const{
		return expression().nnz();
	}
	/// \brief Array of values of the nonzero elements.
	const_pointer values()const{
		return expression().values();
	}
	
	/// \brief Array of indices of the nonzero elements.
	///
	/// Note that there is a pair of indices needed:
	/// When accessing the j-th element in row i you have to write code like this:
	/// index_type start = outer_indices()[i] //aquire start of the i-th row
	/// index = inner_indices()[start+j];
	/// All elements in the row are contained in the range [outer_indices()[i],outer_indices_end()[i])
	/// there might be gaps between the end of the one line and the start of the next!
	index_pointer inner_indices()const{
		return expression().inner_indices();
	}
	
	///\brief Returns an array containing the start of the rows.
	///
	/// See documentation of inner_indices() for more details.
	index_pointer outer_indices()const{
		return expression().outer_indices();
	}
	
	///\brief Returns an array containing the end of the rows.
	///
	/// See documentation of inner_indices() for more details.
	index_pointer outer_indices_end()const{
		return expression().outer_indices_end();
	}
	
	/// \brief Returns the number of nonzero elements in the i-th row/column.
	size_type inner_nnz(index_type i) const {
		return expression().inner_nnz(i);
	}
	
	///\brief Returns the pointer to the beginning of the matrix storage
	///
	/// Grants low-level access to the matrix internals. Element order depends on whether the matrix is row_major or column_major.
	/// to access element (i,j) use storage()[i*stride1()+j*stride2()].
	pointer storage()const{
		return expression().storage();
	}

	// ---------
	// High level interface
	// ---------

	// Element access
	reference operator()(index_type i, index_type j)const{
		return expression()(j, i);
	}
	
	void set_element(size_type i, size_type j,value_type t){
		expression().set_element(j,i,t);
	}

	typedef typename matrix_closure_type::const_column_iterator const_row_iterator;
	typedef typename matrix_closure_type::column_iterator row_iterator;
	typedef typename matrix_closure_type::const_row_iterator const_column_iterator;
	typedef typename matrix_closure_type::row_iterator column_iterator;

	//iterators
	const_row_iterator row_begin(index_type i) const {
		return expression().column_begin(i);
	}
	const_row_iterator row_end(index_type i) const {
		return expression().column_end(i);
	}
	const_column_iterator column_begin(index_type j) const {
		return expression().row_begin(j);
	}
	const_column_iterator column_end(index_type j) const {
		return expression().row_end(j);
	}

	row_iterator row_begin(index_type i) {
		return expression().column_begin(i);
	}
	row_iterator row_end(index_type i) {
		return expression().column_end(i);
	}
	column_iterator column_begin(index_type j) {
		return expression().row_begin(j);
	}
	column_iterator column_end(index_type j) {
		return expression().row_end(j);
	}
	
	typedef typename major_iterator<matrix_transpose<M> >::type major_iterator;
	
	major_iterator set_element(major_iterator pos, index_type index, value_type value){
		return expression().set_element(pos,index,value);
	}
	
	major_iterator clear_range(major_iterator start, major_iterator end){
		return expression().clear_range(start,end);
	}
	
	major_iterator clear_element(major_iterator elem){
		return expression().clear_element(elem);
	}
	
	void clear(){
		expression().clear();
	}
	
	void reserve(size_type non_zeros) {
		expression().reserve(non_zeros);
	}
	
	void reserve_row(size_type row, size_type non_zeros) {
		expression().reserve_row(row,non_zeros);
	}
	void reserve_column(size_type column, size_type non_zeros) {
		expression().reserve_column(column,non_zeros);
	}
	
	// Assignment
	//we implement it by using the identity A^T op= B <=> A op= B^T where op= is one of =,-=,+=
	matrix_transpose& operator = (matrix_transpose const& m) {
		expression() = m.expression();
		return *this;
	}
	template<class E>
	matrix_transpose& operator = (matrix_expression<E> const& e) {
		expression() = matrix_transpose<E const>(e());
		return *this;
	}
private:
	matrix_closure_type m_expression;
};


// (trans m) [i] [j] = m [j] [i]
template<class M>
matrix_transpose<typename const_expression<M>::type >
trans(matrix_expression<M> const& m) {
	return matrix_transpose<typename const_expression<M>::type>(m());
}
template<class M>
temporary_proxy< matrix_transpose<M> > trans(matrix_expression<M>& m) {
	return matrix_transpose<M>(m());
}

template<class M>
temporary_proxy< matrix_transpose<M> > trans(temporary_proxy<M> m) {
	return trans(static_cast<M&>(m));
}

template<class M>
class matrix_row: public vector_expression<matrix_row<M> > {
public:
	typedef M matrix_type;
	typedef std::size_t size_type;
	typedef typename M::difference_type difference_type;
	typedef typename M::value_type value_type;
	typedef typename M::scalar_type scalar_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::const_pointer const_pointer;
	typedef typename pointer<M>::type pointer;

	typedef typename M::index_type index_type;
	typedef typename M::const_index_pointer const_index_pointer;
	typedef typename index_pointer<M>::type index_pointer;

	typedef typename closure<M>::type matrix_closure_type;
	typedef matrix_row<typename const_expression<M>::type> const_closure_type;
	typedef matrix_row<M> closure_type;
	typedef typename M::storage_category storage_category;
	typedef elementwise_tag evaluation_category;

	// Construction and destruction
	matrix_row(matrix_closure_type const& expression, index_type i):m_expression(expression), m_i(i) {
		SIZE_CHECK (i < expression.size1());
	}
	
	template<class E>
	matrix_row(matrix_row<E> const& other)
	:m_expression(other.expression()),m_i(other.index()){}
	
	matrix_closure_type const& expression() const {
		return m_expression;
	}
	matrix_closure_type& expression() {
		return m_expression;
	}
	
	index_type index() const {
		return m_i;
	}
	
	///\brief Returns the size of the vector
	size_type size() const {
		return expression().size2();
	}

	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the stride in memory between two elements
	difference_type stride()const{
		return expression().stride2();
	}
	
	///\brief Returns the pointer to the beginning of the vector storage
	///
	/// Grants low-level access to the vector internals.
	/// to access element i use storage()[i*stride()].
	pointer storage()const{
		return expression().storage()+index()*expression().stride1();
	}
	
	// ---------
	// Sparse low level interface
	// ---------
	
	/// \brief Number of nonzero elements of the vector.
	size_type nnz()const{
		return expression().inner_nnz(m_i);
	}
	/// \brief Array of values of the nonzero elements.
	const_pointer values()const{
		return expression().values()+expression().outer_indices()[m_i];
	}
	
	/// \brief Array of indices of the nonzero elements.
	index_pointer indices()const{
		return expression().inner_indices()+expression().outer_indices()[m_i];
	}
	
	// ---------
	// High level interface
	// ---------
	
	// Element access
	reference operator()(index_type j) const {
		return m_expression(m_i, j);
	}
	reference operator [](index_type j) const {
		return (*this)(j);
	}
	
	void set_element(size_type j,value_type t){
		expression().set_element(m_i,j,t);
	}

	// Assignment
	
	template<class E>
	matrix_row& operator = (vector_expression<E> const& e) {
		return assign(*this, typename vector_temporary<M>::type(e));
	}
	matrix_row& operator = (matrix_row const& e) {
		return assign(*this, typename vector_temporary<M>::type(e));
	}
	
	// Iterator types
	typedef typename M::const_row_iterator const_iterator;
	typedef typename row_iterator<M>::type iterator;

	iterator begin() {
		return expression().row_begin(m_i);
	}
	iterator end() {
		return expression().row_end(m_i);
	}
	const_iterator begin()const{
		return expression().row_begin(m_i);
	}
	const_iterator end()const{
		return expression().row_end(m_i);
	}
	
	iterator set_element(iterator pos, index_type index, value_type value) {
		return set_element(pos, index, value, 
			typename M::orientation(), typename  iterator::iterator_category()
		);
	}
	
	iterator clear_range(iterator start, iterator end) {
		return clear_range(start,end,
			typename M::orientation(), typename  iterator::iterator_category()
		);
	}

	iterator clear_element(iterator pos) {
		return clear_element(pos, 
			typename M::orientation(), typename  iterator::iterator_category()
		);
	}
	
	void clear(){
		clear_range(begin(),end());
	}
	
	void reserve(size_type non_zeros) {
		expression().reserve_row(m_i,non_zeros);
	}
	
private:
	//we need two implementations of the sparse-interface, 
	//depending on whether M is row or column major.
	
	//row major case is trivial
	template<class Tag>
	iterator set_element(iterator pos, index_type index, value_type value, row_major, Tag) {
		return expression().set_element(pos,index,value);
	}
	template<class Tag>
	iterator clear_range(iterator start, iterator end, row_major, Tag) {
		return expression().clear_range(start,end);
	}
	template<class Tag>
	iterator clear_element(iterator pos, row_major, Tag) {
		return expression().clear_element(pos);
	}
	//dense row major case
	iterator set_element(iterator pos, index_type index, value_type value, 
		column_major, 
		dense_random_access_iterator_tag
	) {
		RANGE_CHECK(pos.index() == index);
		*pos = value;
		return pos;
	}
	
	iterator clear_element(iterator pos,
		column_major m, 
		dense_random_access_iterator_tag t
	) {
		return set_element(pos,pos.index(),value_type(),m,t);
	}
	iterator clear_range(iterator start, iterator end, 
		column_major m, 
		dense_random_access_iterator_tag t
	) {
		for(;start != end; ++start)
			clear_element(start,m,t);
		return end;
	}
	//todo: sparse column major case.

	matrix_closure_type m_expression;
	size_type m_i;
};

// Projections
template<class M>
temporary_proxy< matrix_row<M> > row(matrix_expression<M>& expression, typename M::index_type i) {
	return matrix_row<M> (expression(), i);
}
template<class M>
matrix_row<typename const_expression<M>::type>
row(matrix_expression<M> const& expression, typename M::index_type i) {
	return matrix_row<typename const_expression<M>::type> (expression(), i);
}

template<class M>
temporary_proxy<matrix_row<M> > row(temporary_proxy<M> expression, typename M::index_type i) {
	return row(static_cast<M&>(expression), i);
}

template<class M>
class matrix_column: public vector_expression<matrix_column<M> > {
public:
	typedef M matrix_type;
	typedef std::size_t size_type;
	typedef typename M::difference_type difference_type;
	typedef typename M::value_type value_type;
	typedef typename M::scalar_type scalar_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::const_pointer const_pointer;
	typedef typename pointer<M>::type pointer;

	typedef typename M::index_type index_type;
	typedef typename M::const_index_pointer const_index_pointer;
	typedef typename index_pointer<M>::type index_pointer;

	typedef typename closure<M>::type matrix_closure_type;
	typedef matrix_column<typename const_expression<M>::type> const_closure_type;
	typedef matrix_column<M> closure_type;
	typedef typename M::storage_category storage_category;
	typedef elementwise_tag evaluation_category;

	// Construction and destruction
	matrix_column(matrix_closure_type const& expression, index_type j)
	:m_expression(expression), m_j(j) {
		SIZE_CHECK (j < expression.size2());
	}
	
	template<class E>
	matrix_column(matrix_column<E> const& other)
	:m_expression(other.expression()),m_j(other.index()){}
	
	matrix_closure_type const& expression() const {
		return m_expression;
	}
	matrix_closure_type& expression() {
		return m_expression;
	}
	
	index_type index() const {
		return m_j;
	}
	
	///\brief Returns the size of the vector
	size_type size() const {
		return expression().size1();
	}

	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the stride in memory between two elements
	difference_type stride()const{
		return expression().stride1();
	}
	
	///\brief Returns the pointer to the beginning of the vector storage
	///
	/// Grants low-level access to the vector internals.
	/// to access element i use storage()[i*stride()].
	pointer storage()const{
		return expression().storage()+index()*expression().stride2();
	}
	
	// ---------
	// Sparse low level interface
	// ---------
	
	//~ /// \brief Number of nonzero elements of the vector.
	//~ size_type nnz()const{
		//~ return expression().inner_nnz(m_i);
	//~ }
	//~ /// \brief Array of values of the nonzero elements.
	//~ const_pointer values()const{
		//~ return expression().values()+expression().outer_indices()[m_i];
	//~ }
	
	//~ /// \brief Array of indices of the nonzero elements.
	//~ index_pointer indices()const{
		//~ return expression().inner_indices()+expression().outer_indices()[m_i];
	//~ }
	
	// ---------
	// High level interface
	// ---------
	
	// Element access
	reference operator()(index_type i) const {
		return m_expression(i,m_j);
	}
	reference operator [](index_type i) const {
		return (*this)(i);
	}
	
	void set_element(size_type i,value_type t){
		expression().set_element(i,m_j,t);
	}

	// Assignment
	
	template<class E>
	matrix_column& operator = (vector_expression<E> const& e) {
		return assign(*this, typename vector_temporary<M>::type(e));
	}
	matrix_column& operator = (matrix_column const& e) {
		return assign(*this, typename vector_temporary<M>::type(e));
	}
	
	// Iterator types
	typedef typename M::const_column_iterator const_iterator;
	typedef typename column_iterator<M>::type iterator;

	iterator begin() {
		return expression().column_begin(m_j);
	}
	iterator end() {
		return expression().column_end(m_j);
	}
	const_iterator begin()const{
		return expression().column_begin(m_j);
	}
	const_iterator end()const{
		return expression().column_end(m_j);
	}
	
	iterator set_element(iterator pos, index_type index, value_type value) {
		return set_element(pos, index, value, 
			typename M::orientation(), typename  iterator::iterator_category()
		);
	}
	
	iterator clear_range(iterator start, iterator end) {
		return clear_range(start,end,
			typename M::orientation(), typename  iterator::iterator_category()
		);
	}

	iterator clear_element(iterator pos) {
		return clear_element(pos, 
			typename M::orientation(), typename  iterator::iterator_category()
		);
	}
	
	void clear(){
		clear_range(begin(),end());
	}
	
	//~ void reserve(size_type non_zeros) {
		//~ expression().reserve_row(m_i,non_zeros);
	//~ }
	
private:
	//we need two implementations of the sparse-interface, 
	//depending on whether M is row or column major.
	
	//column major case is trivial
	template<class Tag>
	iterator set_element(iterator pos, index_type index, value_type value, column_major, Tag) {
		return expression().set_element(pos,index,value);
	}
	template<class Tag>
	iterator clear_range(iterator start, iterator end, column_major, Tag) {
		return expression().clear_range(start,end);
	}
	template<class Tag>
	iterator clear_element(iterator pos, column_major, Tag) {
		return expression().clear_element(pos);
	}
	//dense row major case
	iterator set_element(iterator pos, index_type index, value_type value, 
		row_major, 
		dense_random_access_iterator_tag
	) {
		RANGE_CHECK(pos.index() == index);
		*pos = value;
		return pos;
	}
	
	iterator clear_element(iterator pos,
		row_major m, 
		dense_random_access_iterator_tag t
	) {
		return set_element(pos,pos.index(),value_type(),m,t);
	}
	iterator clear_range(iterator start, iterator end, 
		row_major m, 
		dense_random_access_iterator_tag t
	) {
		for(;start != end; ++start)
			clear_element(start,m,t);
		return end;
	}
	//todo: sparse column major case.

	matrix_closure_type m_expression;
	size_type m_j;
};

// Projections
template<class M>
temporary_proxy<matrix_column<M> > column(matrix_expression<M>& expression, typename M::index_type j) {
	return matrix_column<M> (expression(), j);
}
template<class M>
matrix_column<typename const_expression<M>::type>
column(matrix_expression<M> const& expression, typename M::index_type j) {
	return matrix_column<typename const_expression<M>::type> (expression(), j);
}

template<class M>
temporary_proxy<matrix_column<M> > column(temporary_proxy<M> expression, typename M::index_type j) {
	return column(static_cast<M&>(expression), j);
}

// Matrix based vector range class representing (off-)diagonals of a matrix.
template<class M>
class matrix_vector_range: public vector_expression<matrix_vector_range<M> > {
public:
	typedef M matrix_type;
	typedef std::size_t size_type;
	typedef typename M::difference_type difference_type;
	typedef typename M::value_type value_type;
	typedef typename M::scalar_type scalar_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::const_pointer const_pointer;
	typedef typename pointer<M>::type pointer;

	typedef typename M::index_type index_type;
	typedef typename M::const_index_pointer const_index_pointer;
	typedef typename index_pointer<M>::type index_pointer;

	typedef typename closure<M>::type matrix_closure_type;
	typedef matrix_vector_range<typename const_expression<M>::type> const_closure_type;
	typedef matrix_vector_range<M>  closure_type;
	typedef typename M::storage_category storage_category;
	typedef elementwise_tag evaluation_category;

	// Construction and destruction
	matrix_vector_range(matrix_type& expression, range const&r1, range const&r2):
		m_expression(expression), m_range1(r1), m_range2(r2) {
		SIZE_CHECK (m_range1.start() <= expression.size1());
		SIZE_CHECK (m_range1.start() + m_range1.size () <= expression.size1());
		SIZE_CHECK (m_range2.start() <= expression.size2());
		SIZE_CHECK (m_range2.start() + m_range2.size() <= expression.size2());
		SIZE_CHECK (m_range1.size() == m_range2.size());
	}
	
	template<class E>
	matrix_vector_range(matrix_vector_range<E> const& other)
	:m_expression(other.expression()),m_range1(other.range1()),m_range2(other.range2()){}

	// Accessors
	size_type start1() const {
		return m_range1.start();
	}
	size_type start2() const {
		return m_range2.start();
	}
	
	const matrix_closure_type& expression() const {
		return m_expression;
	}
	matrix_closure_type& expression() {
		return m_expression;
	}
	
	range const& range1() const{
		return m_range1;
	}
	range const& range2() const{
		return m_range2;
	}
	
	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the size of the vector
	size_type size() const {
		return m_range1.size();
	}
	
	///\brief Returns the stride in memory between two elements
	difference_type stride()const{
		return expression().stride1()+expression().stride2();
	}
	
	void set_element(size_type i,value_type t){
		expression().set_element(m_range1(i),m_range2(i),t);
	}

	
	///\brief Returns the pointer to the beginning of the vector storage
	///
	/// Grants low-level access to the vector internals.
	/// to access element i use storage()[i*stride()].
	pointer storage()const{
		return expression().storage()+start1()*expression().stride1()+start2()*expression().stride1();
	}
	
	// ---------
	// High level interface
	// ---------

	// Element access
	reference operator()(index_type i) const {
		return m_expression(m_range1(i), m_range2(i));
	}
	reference operator [](index_type i) const {
		return (*this)(i);
	}

	// Assignment
	
	template<class E>
	matrix_vector_range& operator = (vector_expression<E> const& e) {
		return assign(*this, typename vector_temporary<M>::type(e));
	}

	typedef indexed_iterator<closure_type> iterator;
	typedef indexed_iterator<const_closure_type> const_iterator;

	// Element lookup
	const_iterator begin()const{
		return const_iterator(*this, 0);
	}
	const_iterator end()const{
		return const_iterator(*this, size());
	}

	iterator begin() {
		return iterator(*this, 0);
	}
	iterator end() {
		return iterator(*this, size());
	}
	
	void reserve(){}
	void reserve_row(size_type, size_type) {}
	void reserve_column(size_type, size_type ){}

private:
	matrix_closure_type m_expression;
	range m_range1;
	range m_range2;
};

///\brief Returns the diagonal of a constant square matrix as vector.
///
/// given a matrix 
/// A = (1 2 3)
///     (4 5 6)
///     (7 8 9)
///
/// the diag operation results in
/// diag(A) = (1,5,9)
template<class M>
matrix_vector_range<typename const_expression<M>::type > diag(matrix_expression<M> const& mat){
	SIZE_CHECK(mat().size1() == mat().size2());
	matrix_vector_range<typename const_expression<M>::type > diagonal(mat(),range(0,mat().size1()),range(0,mat().size1()));
	return diagonal;
}

///\brief Returns the diagonal of a square matrix as vector.
///
/// given a matrix 
/// A = (1 2 3)
///     (4 5 6)
///     (7 8 9)
///
/// the diag operation results in
/// diag(A) = (1,5,9)
template<class M>
temporary_proxy< matrix_vector_range<M> > diag(matrix_expression<M>& mat){
	SIZE_CHECK(mat().size1() == mat().size2());
	matrix_vector_range<M> diagonal(mat(),range(0,mat().size1()),range(0,mat().size1()));
	return diagonal;
}

template<class M>
temporary_proxy< matrix_vector_range<M> > diag(temporary_proxy<M> mat){
	return diag(static_cast<M&>(mat));
}

// Matrix based range class
template<class M>
class matrix_range:public matrix_expression<matrix_range<M> > {
public:
	typedef M matrix_type;
	typedef std::size_t size_type;
	typedef typename M::difference_type difference_type;
	typedef typename M::value_type value_type;
	typedef typename M::scalar_type scalar_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::const_pointer const_pointer;
	typedef typename pointer<M>::type pointer;

	typedef typename M::index_type index_type;
	typedef typename M::const_index_pointer const_index_pointer;
	typedef typename index_pointer<M>::type index_pointer;

	typedef typename closure<M>::type matrix_closure_type;
	typedef matrix_range<typename const_expression<M>::type> const_closure_type;
	typedef matrix_range<M> closure_type;
	typedef typename M::storage_category storage_category;
	typedef elementwise_tag evaluation_category;
	typedef typename M::orientation orientation;

	// Construction and destruction

	matrix_range(matrix_closure_type expression, range const&r1, range const&r2)
	:m_expression(expression), m_range1(r1), m_range2(r2) {
		SIZE_CHECK(r1.start() <= expression.size1());
		SIZE_CHECK(r1.start() +r1.size() <= expression.size1());
		SIZE_CHECK(r2.start() <= expression.size2());
		SIZE_CHECK(r2.start() +r2.size() <= expression.size2());
	}
	
	//conversion closure->const_closure
	template<class E>
	matrix_range(
		matrix_range<E> const& other,
		typename boost::disable_if<
			boost::is_same<E,matrix_range>
		>::type* dummy = 0
	):m_expression(other.expression())
	, m_range1(other.range1())
	, m_range2(other.range2()){}
		
	// Accessors
	size_type start1() const {
		return m_range1.start();
	}
	size_type start2() const {
		return m_range2.start();
	}

	matrix_closure_type expression() const {
		return m_expression;
	}
	matrix_closure_type& expression() {
		return m_expression;
	}
	
	range const& range1() const{
		return m_range1;
	}
	range const& range2() const{
		return m_range2;
	}
	
	// ---------
	// Dense Low level interface
	// ---------
	
	///\brief Returns the number of rows of the matrix.
	size_type size1() const {
		return m_range1.size();
	}
	///\brief Returns the number of columns of the matrix.
	size_type size2() const {
		return m_range2.size();
	}
	
	///\brief Returns the stride in memory between two rows.
	difference_type stride1()const{
		return expression().stride1();
	}
	///\brief Returns the stride in memory between two columns.
	difference_type stride2()const{
		return expression().stride2();
	}
	
	///\brief Returns the pointer to the beginning of the matrix storage
	///
	/// Grants low-level access to the matrix internals. Element order depends on whether the matrix is row_major or column_major.
	/// to access element (i,j) use storage()[i*stride1()+j*stride2()].
	pointer storage()const{
		return expression().storage()+start1()*stride1()+start2()*stride2();
	}
	
	// ---------
	// High level interface
	// ---------
	

	// Element access
	reference operator()(index_type i, index_type j)const{
		return m_expression(m_range1(i), m_range2(j));
	}

	// Assignment
	
	matrix_range& operator = (matrix_range const& e) {
		return assign(*this, typename matrix_temporary<matrix_range>::type(e));
	}
	template<class E>
	matrix_range& operator = (matrix_expression<E> const& e) {
		return assign(*this, typename matrix_temporary<E>::type(e));
	}

	// Iterator types
	typedef subrange_iterator<typename row_iterator<M>::type> row_iterator;
	typedef subrange_iterator<typename column_iterator<M>::type> column_iterator;
	typedef subrange_iterator<typename M::const_row_iterator> const_row_iterator;
	typedef subrange_iterator<typename M::const_column_iterator> const_column_iterator;

	// Element lookup
	const_row_iterator row_begin(index_type i) const {
		return const_row_iterator(
			expression().row_begin(i+start1()),expression().row_end(i+start1()),
			start2(),start2()
		);
	}
	row_iterator row_begin(index_type i){
		return row_iterator(
			expression().row_begin(i+start1()),expression().row_end(i+start1()),
			start2(),start2()
		);
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(
			expression().row_begin(i+start1()),expression().row_end(i+start1()),
			start2()+size2(),start2()
		);
	}
	row_iterator row_end(index_type i){
		return row_iterator(
			expression().row_begin(i+start1()),expression().row_end(i+start1()),
			start2()+size2(),start2()
		);
	}
	const_column_iterator column_begin(index_type j) const {
		return const_column_iterator(
			expression().column_begin(j+start2()),expression().column_end(j+start2()),
			start1(),start1()
		);
	}
	column_iterator column_begin(index_type j) {
		return column_iterator(
			expression().column_begin(j+start2()),expression().column_end(j+start2()),
			start1(),start1()
		);
	}
	const_column_iterator column_end(index_type j) const {
		return const_column_iterator(
			expression().column_begin(j+start2()),expression().column_end(j+start2()),
			start1()+size1(),start1()
		);
	}
	column_iterator column_end(index_type j) {
		return column_iterator(
			expression().column_begin(j+start2()),expression().column_end(j+start2()),
			start1()+size1(),start1()
		);
	}
	typedef typename major_iterator<matrix_range<M> >::type major_iterator;
	
	major_iterator set_element(major_iterator pos, index_type index, value_type value) {
		return expression().set_element(pos.inner(),index+orientation::index_m(start1(),start2()),value);
	}
	
	major_iterator clear_element(major_iterator elem) {
		return major_iterator(expression().clear_element(elem.inner()),orientation::size_m(start1(),start2()));
	}
	
	major_iterator clear_range(major_iterator start, major_iterator end) {
		return major_iterator(expression().clear_range(start.inner(),end.inner()),orientation::size_m(start1(),start2()));
	}
	
	void clear(){
		for(index_type i = 0; i != orientation::index_M(size1(),size2()); ++i)
			clear_range(major_begin(*this,i),major_end(*this,i));
	}
	
	void reserve(size_type){}
	void reserve_row(size_type, size_type) {}
	void reserve_column(size_type, size_type ){}
private:
	matrix_closure_type m_expression;
	range m_range1;
	range m_range2;
};

// Simple Projections
template<class M>
temporary_proxy< matrix_range<M> > subrange(
	matrix_expression<M>& expression, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
) {
	RANGE_CHECK(start1 <= stop1);
	RANGE_CHECK(start2 <= stop2);
	SIZE_CHECK(stop1 <= expression().size1());
	SIZE_CHECK(stop2 <= expression().size2());
	return matrix_range<M> (expression(), range(start1, stop1), range(start2, stop2));
}
template<class M>
matrix_range<typename const_expression<M>::type> subrange(
	matrix_expression<M> const& expression, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
) {
	RANGE_CHECK(start1 <= stop1);
	RANGE_CHECK(start2 <= stop2);
	SIZE_CHECK(stop1 <= expression().size1());
	SIZE_CHECK(stop2 <= expression().size2());
	return matrix_range<typename const_expression<M>::type> (expression(), range(start1, stop1), range(start2, stop2));
}

template<class M>
temporary_proxy< matrix_range<M> > subrange(
	temporary_proxy<M> expression, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
) {
	return subrange(static_cast<M&>(expression),start1,stop1,start2,stop2);
}

template<class M>
temporary_proxy<matrix_range<M> > rows(
	matrix_expression<M>& expression, 
	std::size_t start, std::size_t stop
) {
	RANGE_CHECK(start <= stop);
	SIZE_CHECK(stop <= expression().size1());
	return subrange(expression, start, stop, 0,expression().size2());
}

template<class M>
matrix_range<typename const_expression<M>::type> rows(
	matrix_expression<M> const& expression, 
	std::size_t start, std::size_t stop
) {
	RANGE_CHECK(start <= stop);
	SIZE_CHECK(stop <= expression().size1());
	return subrange(expression, start, stop, 0,expression().size2());
}

template<class M>
temporary_proxy<matrix_range<M> > rows(
	temporary_proxy<M> expression, 
	std::size_t start, std::size_t stop
) {
	return rows(static_cast<M&>(expression),start,stop);
}

template<class M>
temporary_proxy< matrix_range<M> > columns(
	matrix_expression<M>& expression, 
	typename M::index_type start, typename M::index_type stop
) {
	RANGE_CHECK(start <= stop);
	SIZE_CHECK(stop <= expression().size2());
	return subrange(expression, 0,expression().size1(), start, stop);
}

template<class M>
matrix_range<typename const_expression<M>::type> columns(
	matrix_expression<M> const& expression, 
	typename M::index_type start, typename M::index_type stop
) {
	RANGE_CHECK(start <= stop);
	SIZE_CHECK(stop <= expression().size2());
	return subrange(expression, 0,expression().size1(), start, stop);
}

template<class M>
temporary_proxy<matrix_range<M> > columns(
	temporary_proxy<M> expression, 
	std::size_t start, std::size_t stop
) {
	return columns(static_cast<M&>(expression),start,stop);
}

template<class T,class Orientation=row_major>
class dense_matrix_adaptor: public matrix_expression<dense_matrix_adaptor<T,Orientation> > {
	typedef dense_matrix_adaptor<T,Orientation> self_type;
public:

	//std::container types
	typedef typename Orientation::orientation orientation;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename boost::remove_const<T>::type value_type;
	typedef value_type scalar_type;
	typedef value_type const& const_reference;
	typedef T&  reference;
	typedef T* pointer;
	typedef value_type const* const_pointer;

	typedef std::size_t index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type* index_pointer;

	//ublas types
	typedef matrix_reference<self_type const> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef dense_tag storage_category;
	typedef elementwise_tag evaluation_category;
        

	// Construction and destruction
	
	dense_matrix_adaptor(dense_matrix_adaptor<value_type, Orientation> const& expression)
	: m_values(expression.storage())
	, m_size1(expression.size1())
	, m_size2(expression.size2())
	, m_stride1(expression.stride1())
	, m_stride2(expression.stride2())
	{}

	/// \brief Constructor of a vector proxy from a Dense MatrixExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
 	template<class E>
	dense_matrix_adaptor(matrix_expression<E> const& expression)
	: m_values(expression().storage())
	, m_size1(expression().size1())
	, m_size2(expression().size2())
	, m_stride1(expression().stride1())
	, m_stride2(expression().stride2())
	{
		BOOST_STATIC_ASSERT((
			boost::is_same<typename E::orientation,orientation>::value
		));
	}
	
	/// \brief Constructor of a vector proxy from a Dense MatrixExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
 	template<class E>
	dense_matrix_adaptor(matrix_expression<E>& expression)
	: m_values(expression().storage())
	, m_size1(expression().size1())
	, m_size2(expression().size2())
	, m_stride1(expression().stride1())
	, m_stride2(expression().stride2())
	{
		BOOST_STATIC_ASSERT(
			(boost::is_same<typename E::orientation,orientation>::value)
		);
	}
		
	/// \brief Constructor of a vector proxy from a block of memory
	/// \param values the block of memory used
	/// \param size1 size in 1st direction
	/// \param size2 size in 2nd direction
 	/// \param stride1 distance in 1st direction between elements of the self_type in memory
 	/// \param stride2 distance in 2nd direction between elements of the self_type in memory
	dense_matrix_adaptor(
		pointer values, 
		size_type size1, size_type size2,
		difference_type stride1 = 0,difference_type stride2 = 0 
	)
	: m_values(values)
	, m_size1(size1)
	, m_size2(size2)
	, m_stride1(stride1)
	, m_stride2(stride2)
	{
		if(!m_stride1)
			m_stride1= Orientation::stride1(m_size1,m_size2);
		if(!m_stride2)
			m_stride2= Orientation::stride2(m_size1,m_size2);
	}
	
	// ---------
	// Dense low level interface
	// ---------
		
	/// \brief Return the number of rows of the matrix
	size_type size1() const {
		return m_size1;
	}
	/// \brief Return the number of columns of the matrix
	size_type size2() const {
		return m_size2;
	}
	
	difference_type stride1()const{
		return m_stride1;
	}
	difference_type stride2()const{
		return m_stride2;
	}
	
	pointer storage()const{
		return m_values;
	}
	
	// ---------
	// High level interface
	// ---------
	
	// -------
	// ASSIGNING
	// -------
	
	self_type& operator = (self_type const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		return assign(*this, typename matrix_temporary<self_type>::type(e));
	}
	template<class E>
	self_type& operator = (matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		return assign(*this, typename matrix_temporary<self_type>::type(e));
	}
	
	// --------------
	// Element access
	// --------------
	
	const_reference operator () (index_type i, index_type j) const {
		return m_values[i*m_stride1+j*m_stride2];
        }
        reference operator () (index_type i, index_type j) {
		return m_values[i*m_stride1+j*m_stride2];
        }
	void set_element(size_type i, size_type j,value_type t){
		m_values[i*m_stride1+j*m_stride2]  = t;
	}

	
	// --------------
	// ITERATORS
	// --------------


	typedef dense_storage_iterator<T> row_iterator;
	typedef dense_storage_iterator<T> column_iterator;
	typedef dense_storage_iterator<value_type const> const_row_iterator;
	typedef dense_storage_iterator<value_type const> const_column_iterator;

	const_row_iterator row_begin(index_type i) const {
		return const_row_iterator(m_values+i*stride1(),0,stride2());
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(m_values+i*stride1()+size2()*stride2(),size2(),stride2());
	}
	row_iterator row_begin(index_type i){
		return row_iterator(m_values+i*stride1(),0,stride2());
	}
	row_iterator row_end(index_type i){
		return row_iterator(m_values+i*stride1()+size2()*stride2(),size2(),stride2());
	}
	
	const_column_iterator column_begin(index_type j) const {
		return const_column_iterator(m_values+j*stride2(),0,stride1());
	}
	const_column_iterator column_end(index_type j) const {
		return const_column_iterator(m_values+j*stride2()+size1()*stride1(),size1(),stride1());
	}
	column_iterator column_begin(index_type j){
		return column_iterator(m_values+j*stride2(),0,stride1());
	}
	column_iterator column_end(index_type j){
		return column_iterator(m_values+j*stride2()+size1()*stride1(),size1(),stride1());
	}
	
	typedef typename major_iterator<self_type>::type major_iterator;
	
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
	
		
	void clear(){
		for(index_type i = 0; i != size1(); ++i){
			for(index_type j = 0; j != size2(); ++j){
				(*this)(i,j) = value_type();
			}
		}
	}
private:
	pointer m_values;
	size_type m_size1;
	size_type m_size2;
	difference_type m_stride1;
	difference_type m_stride2;
};

/// \brief Converts a chunk of memory into a matrix of given size.
template <class T>
temporary_proxy< dense_matrix_adaptor<T> > adapt_matrix(std::size_t size1, std::size_t size2, T* data){
	return dense_matrix_adaptor<T>(data,size1, size2);
}

/// \brief Converts a 2D C-style array into a matrix of given size.
template <class T, std::size_t M, std::size_t N>
temporary_proxy<dense_matrix_adaptor<T> > adapt_matrix(T (&array)[M][N]){
	return dense_matrix_adaptor<T>(&(array[0][0]),M,N);
}

/// \brief Converts a dense vector to a matrix of a given size
template <class V>
typename boost::enable_if<
	boost::is_same<typename V::storage_category,dense_tag>,
	temporary_proxy< dense_matrix_adaptor<
		typename boost::remove_reference<typename V::reference>::type
	> >
>::type
to_matrix(
	vector_expression<V>& v,
	std::size_t size1, std::size_t size2
){
	typedef typename boost::remove_reference<typename V::reference>::type ElementType;
	return dense_matrix_adaptor<ElementType>(v().storage(), size1, size2);
}

/// \brief Converts a dense vector to a matrix of a given size
template <class V>
typename boost::enable_if<
	boost::is_same<typename V::storage_category,dense_tag>,
	temporary_proxy< dense_matrix_adaptor<typename V::value_type const> >
>::type 
to_matrix(
	vector_expression<V> const& v,
	std::size_t size1, std::size_t size2
){
	return dense_matrix_adaptor<typename V::value_type const>(v().storage(), size1, size2);
}

template <class E>
typename boost::enable_if<
	boost::is_same<typename E::storage_category,dense_tag>,
	temporary_proxy< dense_matrix_adaptor<
		typename boost::remove_reference<typename E::reference>::type
	> >
>::type 
to_matrix(
	temporary_proxy<E> v,
	std::size_t size1, std::size_t size2
){
	return to_matrix(static_cast<E&>(v),size1,size2);
}

}}
#endif
