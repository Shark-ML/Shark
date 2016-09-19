/*!
 * \brief       Classes used for matrix proxies
 * 
 * \author      O. Krause
 * \date        2016
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
 #ifndef SHARK_LINALG_BLAS_MATRIX_PROXY_CLASSES_HPP
#define SHARK_LINALG_BLAS_MATRIX_PROXY_CLASSES_HPP

#include "../assignment.hpp"

namespace shark {
namespace blas {
	
	
///\brief Wraps another expression as a reference.
template<class M>
class matrix_reference:public matrix_expression<matrix_reference<M>, typename M::device_type > {
public:
	typedef typename M::index_type index_type;
	typedef typename M::value_type value_type;
	typedef typename M::scalar_type scalar_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;

	typedef matrix_reference<M const> const_closure_type;
	typedef matrix_reference<M> closure_type;
	typedef typename storage<M>::type storage_type;
	typedef typename M::const_storage_type const_storage_type;
	typedef typename M::evaluation_category evaluation_category;
	typedef typename M::orientation orientation;
	
	// Construction and destruction
	matrix_reference(M& m):m_expression(&m) {}
	template<class E>
	matrix_reference(matrix_reference<E> const& other)
		:m_expression(&other.expression()){}
		
	// Accessors
	M& expression() const {
		return *m_expression;
	}
	M& expression(){
		return *m_expression;
	}
	
	///\brief Returns the number of rows of the matrix.
	index_type size1() const {
		return m_expression->size1();
	}
	///\brief Returns the number of columns of the matrix.
	index_type size2() const {
		return m_expression->size2();
	}
	
	///\brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage()const{
		return m_expression->raw_storage();
	}


	// Element access
	reference operator()(index_type i, index_type j) const {
		return (*m_expression)(i, j);
	}
	
	void set_element(index_type i, index_type j,value_type t){
		m_expression->set_element(i,j,t);
	}


	// Assignment
	template<class E>
	matrix_reference& operator = (matrix_expression<E, typename M::device_type> const& e) {
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
		return m_expression->row_begin(i);
	}
	const_row_iterator row_end(index_type i) const {
		return m_expression->row_end(i);
	}
	row_iterator row_begin(index_type i){
		return m_expression->row_begin(i);
	}
	row_iterator row_end(index_type i){
		return m_expression->row_end(i);
	}
	
	const_column_iterator column_begin(index_type j) const {
		return m_expression->column_begin(j);
	}
	const_column_iterator column_end(index_type j) const {
		return m_expression->column_end(j);
	}
	column_iterator column_begin(index_type j){
		return m_expression->column_begin(j);
	}
	column_iterator column_end(index_type j){
		return m_expression->column_end(j);
	}
	
	row_iterator set_element(row_iterator pos, index_type index, value_type value) {
		return m_expression->set_element(pos,index,value);
	}
	
	row_iterator clear_range(row_iterator start, row_iterator end) {
		return m_expression->clear_range(start,end);
	}
	
	row_iterator clear_element(row_iterator elem) {
		return m_expression->clear_element(elem);
	}
	
	void clear(){
		m_expression->clear();
	}
	
	void reserve(index_type non_zeros) {
		m_expression->reserve(non_zeros);
	}
	
	void reserve_row(index_type row, index_type non_zeros) {
		m_expression->reserve_row(row,non_zeros);
	}
	
	void reserve_column(index_type column, index_type non_zeros) {
		m_expression->reserve_column(column,non_zeros);
	}
	
	
	void swap(matrix_reference& m) {
		m_expression->swap(m.expression());
	}

	friend void swap(matrix_reference& m1, matrix_reference& m2) {
		m1.swap(m2);
	}
	
	void swap_rows(index_type i, index_type j){
		m_expression->swap_rows(i,j);
	}
	
	void swap_columns(index_type i, index_type j){
		m_expression->swap_columns(i,j);
	}
private:
	M* m_expression;
};

/// \brief Matrix transpose.
template<class M>
class matrix_transpose: public matrix_expression<matrix_transpose<M>, typename M::device_type > {
private:
	typedef typename closure<M>::type matrix_closure_type;
public:
	typedef typename M::index_type index_type;
	typedef typename M::value_type value_type;
	typedef typename M::scalar_type scalar_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;

	typedef matrix_transpose<typename const_expression<M>::type> const_closure_type;
	typedef matrix_transpose<M> closure_type;
	typedef typename storage<M>::type storage_type;
	typedef typename M::const_storage_type const_storage_type;
	typedef typename M::evaluation_category evaluation_category;
	typedef typename M::orientation::transposed_orientation orientation;

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
	matrix_closure_type& expression(){
		return m_expression;
	}
	
	///\brief Returns the number of rows of the matrix.
	index_type size1() const {
		return expression().size2();
	}
	///\brief Returns the number of columns of the matrix.
	index_type size2() const {
		return expression().size1();
	}
	
	///\brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage()const{
		return m_expression.raw_storage();
	}

	// ---------
	// High level interface
	// ---------

	// Element access
	reference operator()(index_type i, index_type j)const{
		return expression()(j, i);
	}
	
	void set_element(index_type i, index_type j,value_type t){
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
	
	void reserve(index_type non_zeros) {
		expression().reserve(non_zeros);
	}
	
	void reserve_row(index_type row, index_type non_zeros) {
		expression().reserve_row(row,non_zeros);
	}
	void reserve_column(index_type column, index_type non_zeros) {
		expression().reserve_column(column,non_zeros);
	}
	
	// Assignment
	//we implement it by using the identity A^T op= B <=> A op= B^T where op= is one of =,-=,+=
	matrix_transpose& operator = (matrix_transpose const& m) {
		expression() = m.expression();
		return *this;
	}
	template<class E>
	matrix_transpose& operator = (matrix_expression<E, typename M::device_type> const& e) {
		expression() = matrix_transpose<E const>(e());
		return *this;
	}
private:
	matrix_closure_type m_expression;
};

template<class M>
class matrix_row: public vector_expression<matrix_row<M>, typename M::device_type > {
private:
	typedef typename closure<M>::type matrix_closure_type;
	static_assert((!std::is_same<typename M::evaluation_category::tag,sparse_tag>::value ||
			!boost::is_same<typename M::orientation::orientation, column_major>::value),
			"Can not get row of sparse column major matrix");
public:
	typedef typename M::value_type value_type;
	typedef typename M::scalar_type scalar_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::index_type index_type;

	typedef matrix_row<M> closure_type;
	typedef matrix_row<typename const_expression<M>::type> const_closure_type;
	typedef typename storage<M>::type::row_storage storage_type;
	typedef typename M::const_storage_type::row_storage const_storage_type;
	typedef typename M::evaluation_category evaluation_category;

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
	index_type size() const {
		return expression().size2();
	}
	
	/// \brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage()const{
		return m_expression.raw_storage().row(m_i, typename M::orientation());
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
	
	void set_element(index_type j,value_type t){
		expression().set_element(m_i,j,t);
	}

	// Assignment
	
	template<class E>
	matrix_row& operator = (vector_expression<E,typename M::device_type> const& e) {
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
		return set_element(pos, index, value, typename M::orientation());
	}
	
	iterator clear_range(iterator start, iterator end) {
		return clear_range(start,end,typename M::orientation());
	}

	iterator clear_element(iterator pos) {
		return clear_element(pos, typename M::orientation());
	}
	
	void clear(){
		clear_range(begin(),end());
	}
	
	void reserve(index_type non_zeros) {
		expression().reserve_row(m_i,non_zeros);
	}
	
private:
	
	//row major case is trivial
	iterator set_element(iterator pos, index_type index, value_type value, row_major) {
		return expression().set_element(pos,index,value);
	}
	iterator clear_range(iterator start, iterator end, row_major) {
		return expression().clear_range(start,end);
	}
	iterator clear_element(iterator pos, row_major) {
		return expression().clear_element(pos);
	}
	//dense row major case
	iterator set_element(iterator pos, index_type index, value_type value, column_major ) {
		RANGE_CHECK(pos.index() == index);
		*pos = value;
		return pos;
	}
	
	iterator clear_element(iterator pos,column_major m) {
		return set_element(pos,pos.index(),value_type(),m);
	}
	iterator clear_range(iterator start, iterator end, column_major m) {
		for(;start != end; ++start)
			clear_element(start,m);
		return end;
	}

	matrix_closure_type m_expression;
	index_type m_i;
};

// Matrix based vector range class representing (off-)diagonals of a matrix.
template<class M>
class matrix_vector_range: public vector_expression<matrix_vector_range<M>, typename M::device_type > {
private:
	typedef typename closure<M>::type matrix_closure_type;
public:
	typedef typename M::value_type value_type;
	typedef typename M::scalar_type scalar_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::index_type index_type;

	typedef matrix_vector_range<M> closure_type;
	typedef matrix_vector_range<typename const_expression<M>::type> const_closure_type;
	typedef typename storage<M>::type storage_type;
	typedef typename M::const_storage_type const_storage_type;
	typedef typename M::evaluation_category evaluation_category;

	// Construction and destruction
	matrix_vector_range(matrix_closure_type expression, index_type start1, index_type end1, index_type start2, index_type end2)
	:m_expression(expression), m_start1(start1), m_start2(start2), m_size(end1-start1){
		SIZE_CHECK(start1 <= expression.size1());
		SIZE_CHECK(end1 <= expression.size1());
		SIZE_CHECK(start2 <= expression.size2());
		SIZE_CHECK(end2 <= expression.size2());
		SIZE_CHECK(m_size == end2-start2);
	}
	
	template<class E>
	matrix_vector_range(matrix_vector_range<E> const& other)
	: m_expression(other.expression())
	, m_start1(other.start1())
	, m_start2(other.start2()), m_size(other.size()){}
	
	// Accessors
	index_type start1() const {
		return m_start1;
	}
	index_type start2() const {
		return m_start2;
	}
	
	matrix_closure_type const& expression() const {
		return m_expression;
	}
	matrix_closure_type& expression() {
		return m_expression;
	}
	
	///\brief Returns the size of the vector
	index_type size() const {
		return m_size;
	}
	
	/// \brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage()const{
		return m_expression.raw_storage().diag();
	}
	
	// ---------
	// High level interface
	// ---------

	// Element access
	reference operator()(index_type i) const {
		return m_expression(start1()+i,start2()+i);
	}
	reference operator [](index_type i) const {
		return (*this)(i);
	}
	
	void set_element(index_type i,value_type t){
		expression().set_element(start1()+i,start2()+i,t);
	}

	// Assignment
	
	template<class E>
	matrix_vector_range& operator = (vector_expression<E, typename M::device_type> const& e) {
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
	void reserve_row(index_type, index_type) {}
	void reserve_column(index_type, index_type ){}

private:
	matrix_closure_type m_expression;
	index_type m_start1;
	index_type m_start2;
	index_type m_size;
};

// Matrix based range class
template<class M>
class matrix_range:public matrix_expression<matrix_range<M>, typename M::device_type > {
private:
	typedef typename closure<M>::type matrix_closure_type;
public:
	typedef typename M::value_type value_type;
	typedef typename M::scalar_type scalar_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::index_type index_type;
	
	typedef matrix_range<M> closure_type;
	typedef matrix_range<typename const_expression<M>::type> const_closure_type;
	typedef typename storage<M>::type storage_type;
	typedef typename M::const_storage_type const_storage_type;
	typedef typename M::orientation orientation;
	typedef typename M::evaluation_category evaluation_category;

	// Construction and destruction

	matrix_range(matrix_closure_type expression, index_type start1, index_type end1, index_type start2, index_type end2)
	:m_expression(expression), m_start1(start1), m_size1(end1-start1), m_start2(start2), m_size2(end2-start2){
		SIZE_CHECK(start1 <= expression.size1());
		SIZE_CHECK(end1 <= expression.size1());
		SIZE_CHECK(start2 <= expression.size2());
		SIZE_CHECK(end2 <= expression.size2());
		SIZE_CHECK(start1 <= end1);
		SIZE_CHECK(start2 <= end2);
	}
	
	//conversion closure->const_closure
	template<class E>
	matrix_range(
		matrix_range<E> const& other,
		typename boost::disable_if<
			boost::is_same<E,matrix_range>
		>::type* dummy = 0
	):m_expression(other.expression())
	, m_start1(other.start1()), m_size1(other.size1())
	, m_start2(other.start2()), m_size2(other.size2()){}
		
	// Accessors
	index_type start1() const {
		return m_start1;
	}
	index_type start2() const {
		return m_start2;
	}

	matrix_closure_type expression() const {
		return m_expression;
	}
	matrix_closure_type& expression() {
		return m_expression;
	}
	
	///\brief Returns the number of rows of the matrix.
	index_type size1() const {
		return m_size1;
	}
	///\brief Returns the number of columns of the matrix.
	index_type size2() const {
		return m_size2;
	}
	
	/// \brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage()const{
		return m_expression.raw_storage().sub_region(m_start1, m_start2, typename M::orientation());
	}
	
	// ---------
	// High level interface
	// ---------
	

	// Element access
	reference operator()(index_type i, index_type j)const{
		return m_expression(start1() +i, start2() + j);
	}

	// Assignment
	
	matrix_range& operator = (matrix_range const& e) {
		return assign(*this, typename matrix_temporary<matrix_range>::type(e));
	}
	template<class E>
	matrix_range& operator = (matrix_expression<E, typename M::device_type> const& e) {
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
	
	void reserve(index_type){}
	void reserve_row(index_type, index_type) {}
	void reserve_column(index_type, index_type ){}
private:
	matrix_closure_type m_expression;
	index_type m_start1;
	index_type m_size1;
	index_type m_start2;
	index_type m_size2;
};

template<class T,class Orientation=row_major>
class dense_matrix_adaptor: public matrix_expression<dense_matrix_adaptor<T,Orientation>, cpu_tag > {
	typedef dense_matrix_adaptor<T,Orientation> self_type;
public:
	typedef std::size_t index_type;
	typedef typename boost::remove_const<T>::type value_type;
	typedef value_type scalar_type;
	typedef value_type const& const_reference;
	typedef T& reference;

	typedef matrix_reference<self_type> closure_type;
	typedef matrix_reference<self_type const> const_closure_type;
	typedef dense_matrix_storage<T> storage_type;
	typedef dense_matrix_storage<value_type const> const_storage_type;
        typedef Orientation orientation;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction
	
	dense_matrix_adaptor(dense_matrix_adaptor<value_type, Orientation> const& expression)
	: m_values(expression.m_values)
	, m_size1(expression.size1())
	, m_size2(expression.size2())
	, m_stride1(expression.m_stride1)
	, m_stride2(expression.m_stride2)
	{}

	/// \brief Constructor of a vector proxy from a Dense MatrixExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
 	template<class E>
	dense_matrix_adaptor(matrix_expression<E, cpu_tag> const& expression)
	: m_size1(expression().size1())
	, m_size2(expression().size2())
	{
		auto storage_type = expression().raw_storage();
		m_values = storage_type.values;
		m_stride1 = Orientation::index_M(storage_type.leading_dimension,1);
		m_stride2 = Orientation::index_m(storage_type.leading_dimension,1);
		static_assert(boost::is_same<typename E::orientation,orientation>::value, "matrix orientation mismatch");
	}

	/// \brief Constructor of a vector proxy from a Dense MatrixExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
 	template<class E>
	dense_matrix_adaptor(matrix_expression<E, cpu_tag>& expression)
	: m_size1(expression().size1())
	, m_size2(expression().size2())
	{
		auto storage_type = expression().raw_storage();
		m_values = storage_type.values;
		m_stride1 = Orientation::index_M(storage_type.leading_dimension,1);
		m_stride2 = Orientation::index_m(storage_type.leading_dimension,1);
		static_assert(boost::is_same<typename E::orientation,orientation>::value, "matrix orientation mismatch");
	}
		
	/// \brief Constructor of a vector proxy from a block of memory
	/// \param values the block of memory used
	/// \param size1 size in 1st direction
	/// \param size2 size in 2nd direction
 	/// \param stride1 distance in 1st direction between elements of the self_type in memory
 	/// \param stride2 distance in 2nd direction between elements of the self_type in memory
	dense_matrix_adaptor(
		T* values, 
		index_type size1, index_type size2,
		index_type stride1 = 0, index_type stride2 = 0 
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
	index_type size1() const {
		return m_size1;
	}
	/// \brief Return the number of columns of the matrix
	index_type size2() const {
		return m_size2;
	}
	
	///\brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage()const{
		return {m_values, orientation::index_M(m_stride1,m_stride2)};
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
	self_type& operator = (matrix_expression<E, cpu_tag> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		return assign(*this, typename matrix_temporary<self_type>::type(e));
	}
	
	// --------------
	// Element access
	// --------------
	
	const_reference operator() (index_type i, index_type j) const {
		return m_values[i*m_stride1+j*m_stride2];
        }
        reference operator() (index_type i, index_type j) {
		return m_values[i*m_stride1+j*m_stride2];
        }
	void set_element(index_type i, index_type j,value_type t){
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
		return const_row_iterator(m_values+i*m_stride1,0,m_stride2);
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(m_values+i*m_stride1+size2()*m_stride2,size2(),m_stride2);
	}
	row_iterator row_begin(index_type i){
		return row_iterator(m_values+i*m_stride1,0,m_stride2);
	}
	row_iterator row_end(index_type i){
		return row_iterator(m_values+i*m_stride1+size2()*m_stride2,size2(),m_stride2);
	}
	
	const_column_iterator column_begin(index_type j) const {
		return const_column_iterator(m_values+j*m_stride2,0,m_stride1);
	}
	const_column_iterator column_end(index_type j) const {
		return const_column_iterator(m_values+j*m_stride2+size1()*m_stride1,size1(),m_stride1);
	}
	column_iterator column_begin(index_type j){
		return column_iterator(m_values+j*m_stride2,0,m_stride1);
	}
	column_iterator column_end(index_type j){
		return column_iterator(m_values+j*m_stride2+size1()*m_stride1,size1(),m_stride1);
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
	T* m_values;
	index_type m_size1;
	index_type m_size2;
	index_type m_stride1;
	index_type m_stride2;
};



}}
#endif
