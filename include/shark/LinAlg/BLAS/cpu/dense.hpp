/*!
 * \brief       Dense Matrix and Vector classes
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
#ifndef REMORA_CPU_DENSE_HPP
#define REMORA_CPU_DENSE_HPP

#include "iterator.hpp"
#include "../detail/proxy_optimizers_fwd.hpp"
#include "../assignment.hpp"


#include <initializer_list>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>

namespace remora{
	
/// \brief Represents a given chunk of memory as a dense vector of elements of type T.
///
/// This adaptor is read/write if T is non-const and read-only if T is const.
template<class T, class Tag>
class dense_vector_adaptor<T, Tag, cpu_tag>: public vector_expression<dense_vector_adaptor<T, Tag, cpu_tag>, cpu_tag > {
public:

	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type const& const_reference;
	typedef T&  reference;

	typedef dense_vector_adaptor<T const, Tag, cpu_tag> const_closure_type;
	typedef dense_vector_adaptor closure_type;
	typedef dense_vector_storage<T, Tag> storage_type;
	typedef dense_vector_storage<value_type const, Tag> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction

	/// \brief Constructor of a vector proxy from a vector
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression The Expression from which to construct the Proxy
	dense_vector_adaptor(vector<value_type, cpu_tag> const& expression)
	: m_values(expression.raw_storage().values)
	, m_size(expression.size())
	, m_stride(expression.raw_storage().stride){}
	
	/// \brief Constructor of a vector proxy from a vector
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression The Expression from which to construct the Proxy
	dense_vector_adaptor(vector<value_type, cpu_tag>& expression)
	: m_values(expression.raw_storage().values)
	, m_size(expression.size())
	, m_stride(expression.raw_storage().stride){}
		
	/// \brief Constructor of a vector proxy from a block of memory
	/// \param values the block of memory used
	/// \param size size of the vector
	/// \param stride distance between elements of the vector in memory
	dense_vector_adaptor(T* values, size_type size, size_type stride = 1 ):
		m_values(values),m_size(size),m_stride(stride){}

	
	dense_vector_adaptor(storage_type const& storage, no_queue, size_type size):
		m_values(storage.values),m_size(size),m_stride(storage.stride){}	

		
	/// \brief Copy-constructor of a vector
	/// \param v is the proxy to be copied
	template<class U, class Tag2>
	dense_vector_adaptor(dense_vector_adaptor<U, Tag2> const& v)
	: m_values(v.raw_storage().values)
	, m_size(v.size())
	, m_stride(v.raw_storage().stride){
		static_assert(std::is_convertible<Tag2,Tag>::value, "Can not convert storage type of argument to the given Tag");
	}
	
	dense_vector_adaptor& operator = (dense_vector_adaptor const& e) {
		return assign(*this, typename vector_temporary<dense_vector_adaptor>::type(e));
	}
	template<class E>
	dense_vector_adaptor& operator = (vector_expression<E, cpu_tag> const& e) {
		return assign(*this, typename vector_temporary<E>::type(e));
	}
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_size;
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage() const{
		return {m_values,m_stride};
	}
	
	typename device_traits<cpu_tag>::queue_type& queue() const{
		return device_traits<cpu_tag>::default_queue();
	}
	// --------------
	// Element access
	// --------------

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator()(size_type i) const {
		return m_values[i*m_stride];
	}
	
	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator()(size_type i) {
		return m_values[i*m_stride];
	}	

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator[](size_type i) const {
		return m_values[i*m_stride];
	}
	
	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator[](size_type i) {
		return m_values[i*m_stride];
	}
	
	void clear(){
		std::fill(begin(), end(), value_type/*zero*/());
	}

	// --------------
	// ITERATORS
	// --------------

	typedef iterators::dense_storage_iterator<T> iterator;
	typedef iterators::dense_storage_iterator<value_type const> const_iterator;

	/// \brief return an iterator on the first element of the vector
	const_iterator begin() const {
		return const_iterator(m_values, 0, m_stride);
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator end() const {
		return const_iterator(m_values + size() * m_stride, size(), m_stride);
	}

	/// \brief Return an iterator on the first element of the vector
	iterator begin(){
		return iterator(m_values, 0, m_stride);
	}

	/// \brief Return an iterator at the end of the vector
	iterator end(){
		return iterator(m_values + size() * m_stride, size(), m_stride);
	}
	
private:
	dense_vector_adaptor(vector<value_type, cpu_tag>&& expression); // no construction from temporaries
	T* m_values;
	size_type m_size;
	size_type m_stride;
};
	

template<class T,class Orientation, class Tag>
class dense_matrix_adaptor<T,Orientation,Tag, cpu_tag>: public matrix_expression<dense_matrix_adaptor<T,Orientation, Tag, cpu_tag>, cpu_tag > {
public:
	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type result_type;
	typedef value_type const& const_reference;
	typedef T& reference;

	typedef dense_matrix_adaptor<T,Orientation, Tag, cpu_tag> closure_type;
	typedef dense_matrix_adaptor<value_type const,Orientation, Tag, cpu_tag> const_closure_type;
	typedef dense_matrix_storage<T,Tag> storage_type;
	typedef dense_matrix_storage<value_type const,Tag> const_storage_type;
	typedef Orientation orientation;
	typedef elementwise<dense_tag> evaluation_category;

	template<class,class,class,class> friend class dense_matrix_adaptor;

	// Construction and destruction
	template<class U, class TagU>
	dense_matrix_adaptor(dense_matrix_adaptor<U, Orientation, TagU, cpu_tag> const& expression)
	: m_values(expression.m_values)
	, m_size1(expression.size1())
	, m_size2(expression.size2())
	, m_leading_dimension(expression.m_leading_dimension)
	{static_assert(std::is_convertible<TagU,Tag>::value, "Can not convert storage type of argument to the given Tag");}
	
	dense_matrix_adaptor(storage_type const& storage, no_queue, std::size_t size1, std::size_t size2)
	: m_values(storage.values)
	, m_size1(size1)
	, m_size2(size2)
	, m_leading_dimension(storage.leading_dimension){}

	/// \brief Constructor of a vector proxy from a Dense matrix
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
	dense_matrix_adaptor(matrix<value_type, Orientation, cpu_tag> const& expression)
	: m_size1(expression.size1())
	, m_size2(expression.size2())
	{
		auto storage = expression.raw_storage();
		m_values = storage.values;
		m_leading_dimension = storage.leading_dimension;
	}

	/// \brief Constructor of a vector proxy from a Dense matrix
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
	dense_matrix_adaptor(matrix<value_type, Orientation, cpu_tag>& expression)
	: m_size1(expression.size1())
	, m_size2(expression.size2()){
		auto storage = expression.raw_storage();
		m_values = storage.values;
		m_leading_dimension = storage.leading_dimension;
	}
		
	/// \brief Constructor of a vector proxy from a block of memory
	/// \param values the block of memory used
	/// \param size1 size in 1st direction
	/// \param size2 size in 2nd direction
	/// \param leading_dimension distance between two elements in the "slow" direction. 
	dense_matrix_adaptor(
		T* values, 
		size_type size1, size_type size2,
		size_type leading_dimension = 0
	)
	: m_values(values)
	, m_size1(size1)
	, m_size2(size2)
	, m_leading_dimension(leading_dimension)
	{
		if(!m_leading_dimension)
			m_leading_dimension = orientation::index_m(m_size1,m_size2);
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
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage()const{
		return {m_values, m_leading_dimension};
	}
	
	typename device_traits<cpu_tag>::queue_type& queue()const{
		return device_traits<cpu_tag>::default_queue();
	}
	
	// ---------
	// High level interface
	// ---------
	
	// -------
	// ASSIGNING
	// -------
	
	dense_matrix_adaptor& operator = (dense_matrix_adaptor const& e) {
		REMORA_SIZE_CHECK(size1() == e.size1());
		REMORA_SIZE_CHECK(size2() == e.size2());
		return assign(*this, typename matrix_temporary<dense_matrix_adaptor>::type(e));
	}
	template<class E>
	dense_matrix_adaptor& operator = (matrix_expression<E, cpu_tag> const& e) {
		REMORA_SIZE_CHECK(size1() == e().size1());
		REMORA_SIZE_CHECK(size2() == e().size2());
		return assign(*this, typename matrix_temporary<dense_matrix_adaptor>::type(e));
	}
	template<class E>
	dense_matrix_adaptor& operator = (vector_set_expression<E, cpu_tag> const& e) {
		REMORA_SIZE_CHECK(size1() == typename E::point_orientation::index_M(e().size(), e().point_size()));
		REMORA_SIZE_CHECK(size2() == typename E::point_orientation::index_M(e().size(), e().point_size()));
		return assign(*this, typename matrix_temporary<dense_matrix_adaptor>::type(e));
	}
	
	// --------------
	// Element access
	// --------------
	
	reference operator() (size_type i, size_type j) const {
		REMORA_SIZE_CHECK( i < m_size1);
		REMORA_SIZE_CHECK( j < m_size2);
		return m_values[orientation::element(i, j, m_leading_dimension)];
	}
	void set_element(size_type i, size_type j,value_type t){
		REMORA_SIZE_CHECK( i < m_size1);
		REMORA_SIZE_CHECK( j < m_size2);
		return m_values[orientation::element(i, j, m_leading_dimension)];
	}

	// --------------
	// ITERATORS
	// --------------

	typedef iterators::dense_storage_iterator<T> major_iterator;
	typedef iterators::dense_storage_iterator<value_type const> const_major_iterator;

	const_major_iterator major_begin(size_type i) const {
		return const_major_iterator(m_values + i * m_leading_dimension, 0, 1);
	}
	const_major_iterator major_end(size_type i) const {
		return const_major_iterator(m_values + i * m_leading_dimension + minor_size(*this), minor_size(*this), 1);
	}
	major_iterator major_begin(size_type i){
		return major_iterator(m_values + i * m_leading_dimension, 0, 1);
	}
	major_iterator major_end(size_type i){
		return major_iterator(m_values + i * m_leading_dimension + minor_size(*this), minor_size(*this), 1);
	}
	
	void swap_rows(size_type i, size_type j){
		for(std::size_t k = 0; k != size2(); ++k){
			std::swap((*this)(i,k),(*this)(j,k));
		}
	}
	
	void swap_columns(size_type i, size_type j){
		for(std::size_t k = 0; k != size1(); ++k){
			std::swap((*this)(k,i),(*this)(k,j));
		}
	}
	
	major_iterator set_element(major_iterator pos, size_type index, value_type value) {
		REMORA_RANGE_CHECK(index == pos.index());
		*pos = value;
		//return the iterator to the next element
		return pos + 1;
	}
	
		
	void clear(){
		for(size_type i = 0; i != major_size(*this); ++i){
			for(size_type j = 0; j != minor_size(*this); ++j){
				m_values[i * m_leading_dimension + j] = value_type();
			}
		}
	}
private:
	dense_matrix_adaptor(matrix<value_type, Orientation, cpu_tag>&& expression); //no construction from temporary matrix
	T* m_values;
	size_type m_size1;
	size_type m_size2;
	size_type m_leading_dimension;
};


/** \brief A dense matrix of values of type \c T.
 *
 * For a \f$(m \times n)\f$-dimensional matrix and \f$ 0 \leq i < m, 0 \leq j < n\f$, every element \f$ m_{i,j} \f$ is mapped to
 * the \f$(i.n + j)\f$-th element of the container for row major orientation or the \f$ (i + j.m) \f$-th element of
 * the container for column major orientation. In a dense matrix all elements are represented in memory in a
 * contiguous chunk of memory by definition.
 *
 * Orientation can also be specified, otherwise a \c major_major is used.
 *
 * \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
 * \tparam L the storage organization. It can be either \c major_major or \c minor_major. Default is \c major_major
 */
template<class T, class L>
class matrix<T,L,cpu_tag>:public matrix_container<matrix<T, L, cpu_tag>, cpu_tag > {
	typedef std::vector<T> array_type;
public:
	typedef typename array_type::value_type value_type;
	typedef typename array_type::const_reference const_reference;
	typedef typename array_type::reference reference;
	typedef typename array_type::size_type size_type;

	typedef dense_matrix_adaptor<T const,L, continuous_dense_tag, cpu_tag> const_closure_type;
	typedef dense_matrix_adaptor<T,L, continuous_dense_tag, cpu_tag> closure_type;
	typedef dense_matrix_storage<T, continuous_dense_tag> storage_type;
	typedef dense_matrix_storage<T const, continuous_dense_tag> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;
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
			REMORA_SIZE_CHECK(pos->size() == m_size2);
			std::copy(pos->begin(),pos->end(),major_begin(i));
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
	matrix(matrix_expression<E, cpu_tag> const& e)
	: m_size1(e().size1())
	, m_size2(e().size2())
	, m_data(m_size1 * m_size2) {
		assign(*this,e);
	}
	
	/// \brief Constructor of a dense matrix from a vector-set expression.
	/// 
	/// Constructs the matrix by evaluating the expression and assigning the
	/// results to the newly constructed matrix using a call to assign.
	///
	/// \param e is a vector set expression
	template<class E>
	matrix(vector_set_expression<E, cpu_tag> const& e)
	: m_size1(E::point_orientation::index_M(e().size(), e().point_size()))
	, m_size2(E::point_orientation::index_m(e().size(), e().point_size()))
	, m_data(m_size1 * m_size2) {
		assign(*this,e().expression());
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
	matrix& operator = (matrix_container<C, cpu_tag> const& m) {
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
	matrix& operator = (matrix_expression<E, cpu_tag> const& e) {
		matrix temporary(e);
		swap(temporary);
		return *this;
	}
	
	/// \brief Assigns e to this
	/// 
	/// evaluates the vector-set expression and assign the
	/// results to this using a call to assign.
	/// A temporary is created to prevent aliasing.
	///
	/// \param e is a matrix expression
	template<class E>
	matrix& operator = (vector_set_expression<E, cpu_tag> const& e) {
		matrix temporary(e);
		swap(temporary);
		return *this;
	}
	
	///\brief Returns the number of rows of the matrix.
	size_type size1() const {
		return m_size1;
	}
	///\brief Returns the number of columns of the matrix.
	size_type size2() const {
		return m_size2;
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return {m_data.data(), leading_dimension()};
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage()const{
		return {m_data.data(), leading_dimension()};
	}
	typename device_traits<cpu_tag>::queue_type& queue() const{
		return device_traits<cpu_tag>::default_queue();
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
	const_reference operator()(size_type i, size_type j) const {
		REMORA_SIZE_CHECK(i < size1());
		REMORA_SIZE_CHECK(j < size2());
		return m_data[orientation::element(i, j, leading_dimension())];
	}
	reference operator()(size_type i, size_type j) {
		REMORA_SIZE_CHECK(i < size1());
		REMORA_SIZE_CHECK(j < size2());
		return m_data[orientation::element(i, j, leading_dimension())];
	}
	
	void set_element(size_type i, size_type j,value_type t){
		REMORA_SIZE_CHECK(i < size1());
		REMORA_SIZE_CHECK(j < size2());
		m_data[orientation::element(i, j, leading_dimension())]  = t;
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
	
	friend void swap_rows(matrix& a, size_type i, matrix& b, size_type j){
		REMORA_SIZE_CHECK(i < a.size1());
		REMORA_SIZE_CHECK(j < b.size1());
		REMORA_SIZE_CHECK(a.size2() == b.size2());
		for(std::size_t k = 0; k != a.size2(); ++k){
			std::swap(a(i,k),b(j,k));
		}
	}
	
	void swap_rows(size_type i, size_type j) {
		if(i == j) return;
		for(std::size_t k = 0; k != size2(); ++k){
			std::swap((*this)(i,k),(*this)(j,k));
		}
	}
	
	
	friend void swap_columns(matrix& a, size_type i, matrix& b, size_type j){
		REMORA_SIZE_CHECK(i < a.size2());
		REMORA_SIZE_CHECK(j < b.size2());
		REMORA_SIZE_CHECK(a.size1() == b.size1());
		for(std::size_t k = 0; k != a.size1(); ++k){
			std::swap(a(k,i),b(k,j));
		}
	}
	
	void swap_columns(size_type i, size_type j) {
		if(i == j) return;
		for(std::size_t k = 0; k != size1(); ++k){
			std::swap((*this)(k,i),(*this)(k,j));
		}
	}

	//Iterators
	typedef iterators::dense_storage_iterator<T> major_iterator;
	typedef iterators::dense_storage_iterator<value_type const> const_major_iterator;

	const_major_iterator major_begin(size_type i) const {
		return const_major_iterator(m_data.data() + i * leading_dimension(), 0, 1);
	}
	const_major_iterator major_end(size_type i) const {
		return const_major_iterator(m_data.data() + i * leading_dimension() + minor_size(*this), minor_size(*this), 1);
	}
	major_iterator major_begin(size_type i){
		return major_iterator(m_data.data() + i * leading_dimension(), 0, 1);
	}
	major_iterator major_end(size_type i){
		return major_iterator(m_data.data() + i * leading_dimension() + minor_size(*this), minor_size(*this), 1);
	}
	
	major_iterator set_element(major_iterator pos, size_type index, value_type value) {
		REMORA_RANGE_CHECK(index == pos.index());
		*pos = value;
		//return the iterator to the next element
		return pos + 1;
	}

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
	size_type leading_dimension() const {
		return orientation::index_m(m_size1, m_size2);
	}
	
	size_type m_size1;
	size_type m_size2;
	array_type m_data;
};

template<class T>
class vector<T,cpu_tag>: public vector_container<vector<T, cpu_tag>, cpu_tag > {

	typedef std::vector<typename std::conditional<std::is_same<T,bool>::value,char,T>::type > array_type;
public:
	typedef typename array_type::value_type value_type;
	typedef typename array_type::const_reference const_reference;
	typedef typename array_type::reference reference;
	typedef typename array_type::size_type size_type;

	typedef dense_vector_adaptor<T const, continuous_dense_tag, cpu_tag> const_closure_type;
	typedef dense_vector_adaptor<T,continuous_dense_tag, cpu_tag> closure_type;
	typedef dense_vector_storage<value_type, continuous_dense_tag> storage_type;
	typedef dense_vector_storage<value_type const, continuous_dense_tag> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction

	/// \brief Constructor of a vector
	/// By default it is empty, i.e. \c size()==0.
	vector() = default;

	/// \brief Constructor of a vector with a predefined size
	/// By default, its elements are initialized to 0.
	/// \param size initial size of the vector
	explicit vector(size_type size):m_storage(size) {}

	/// \brief Constructor of a vector with a predefined size and a unique initial value
	/// \param size of the vector
	/// \param init value to assign to each element of the vector
	vector(size_type size, const value_type& init):m_storage(size, init) {}

	/// \brief Copy-constructor of a vector
	/// \param v is the vector to be duplicated
	vector(vector const& v) = default;
		
	/// \brief Move-constructor of a vector
	/// \param v is the vector to be moved
	//~ vector(vector && v) = default; //vc++ can not default this. true story
	vector(vector && v): m_storage(std::move(v.m_storage)){}
		
	vector(std::initializer_list<T>  list) : m_storage(list.begin(),list.end()){}
		
	/// \brief Constructs the vector from a predefined range
	template<class Iter>
	vector(Iter begin, Iter end):m_storage(begin,end){}

	/// \brief Copy-constructor of a vector from a vector_expression
	/// \param e the vector_expression whose values will be duplicated into the vector
	template<class E>
	vector(vector_expression<E, cpu_tag> const& e):m_storage(e().size()) {
		assign(*this, e);
	}
	
	// -------------------
	// Assignment operators
	// -------------------
	
	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector). This method does not create any temporary.
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	vector& operator = (vector const& v) = default;
	
	/// \brief Move-Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	//~ vector& operator = (vector && v) = default; //vc++ can not default this. true story
	vector& operator = (vector && v){
		m_storage = std::move(v.m_storage);
		return *this;
	}
	
	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector). This method does not create any temporary.
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	template<class C>          // Container assignment without temporary
	vector& operator = (vector_container<C, cpu_tag> const& v) {
		resize(v().size());
		return assign(*this, v);
	}

	/// \brief Assign the result of a vector_expression to the vector
	/// Assign the result of a vector_expression to the vector.
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vector& operator = (vector_expression<E, cpu_tag> const& e) {
		vector temporary(e);
		swap(*this,temporary);
		return *this;
	}

	// ---------
	// Storage interface
	// ---------
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_storage.size();
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return {m_storage.data(),1};
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return {m_storage.data(),1};
	}
	typename device_traits<cpu_tag>::queue_type& queue() const{
		return device_traits<cpu_tag>::default_queue();
	}
	
	// ---------
	// High level interface
	// ---------

	/// \brief Return the maximum size of the data container.
	/// Return the upper bound (maximum size) on the data container. Depending on the container, it can be bigger than the current size of the vector.
	size_type max_size() const {
		return m_storage.max_size();
	}

	/// \brief Return true if the vector is empty (\c size==0)
	/// \return \c true if empty, \c false otherwise
	bool empty() const {
		return m_storage.empty();
	}

	/// \brief Resize the vector
	/// \param size new size of the vector
	void resize(size_type size) {
		m_storage.resize(size);
	}

	// --------------
	// Element access
	// --------------

	/// \brief Return a const reference to the element \f$i\f$
	/// Return a const reference to the element \f$i\f$. With some compilers, this notation will be faster than \c operator[]
	/// \param i index of the element
	const_reference operator()(size_type i) const {
		REMORA_RANGE_CHECK(i < size());
		return m_storage[i];
	}

	/// \brief Return a reference to the element \f$i\f$
	/// Return a reference to the element \f$i\f$. With some compilers, this notation will be faster than \c operator[]
	/// \param i index of the element
	reference operator()(size_type i) {
		REMORA_RANGE_CHECK(i < size());
		return m_storage[i];
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator [](size_type i) const {
		REMORA_RANGE_CHECK(i < size());
		return m_storage[i];
	}

	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator [](size_type i) {
		REMORA_RANGE_CHECK(i < size());
		return m_storage[i];
	}
	
	///\brief Returns the first element of the vector
	reference front(){
		return m_storage[0];
	}
	///\brief Returns the first element of the vector
	const_reference front()const{
		return m_storage[0];
	}
	///\brief Returns the last element of the vector
	reference back(){
		return m_storage[size()-1];
	}
	///\brief Returns the last element of the vector
	const_reference back()const{
		return m_storage[size()-1];
	}
	
	///\brief resizes the vector by appending a new element to the end. this invalidates storage 
	void push_back(value_type const& element){
		m_storage.push_back(element);
	}

	/// \brief Clear the vector, i.e. set all values to the \c zero value.
	void clear() {
		std::fill(m_storage.begin(), m_storage.end(), value_type/*zero*/());
	}
	
	// Iterator types
	typedef iterators::dense_storage_iterator<value_type> iterator;
	typedef iterators::dense_storage_iterator<value_type const> const_iterator;
	
	/// \brief return an iterator on the first element of the vector
	const_iterator cbegin() const {
		return const_iterator(m_storage.data(),0);
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator cend() const {
		return const_iterator(m_storage.data()+size(),size());
	}

	/// \brief return an iterator on the first element of the vector
	const_iterator begin() const {
		return cbegin();
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator end() const {
		return cend();
	}

	/// \brief Return an iterator on the first element of the vector
	iterator begin(){
		return iterator(m_storage.data(),0);
	}

	/// \brief Return an iterator at the end of the vector
	iterator end(){
		return iterator(m_storage.data()+size(),size());
	}
	
	/// \brief Swap the content of two vectors
	/// \param v1 is the first vector. It takes values from v2
	/// \param v2 is the second vector It takes values from v1
	friend void swap(vector& v1, vector& v2) {
		v1.m_storage.swap(v2.m_storage);
	}
	// -------------
	// Serialization
	// -------------

	/// Serialize a vector into and archive as defined in Boost
	/// \param ar Archive object. Can be a flat file, an XML file or any other stream
	/// \param file_version Optional file version (not yet used)
	template<class Archive>
	void serialize(Archive &ar, const unsigned int file_version) {
		boost::serialization::collection_size_type count(size());
		ar & count;
		if(!Archive::is_saving::value){
			resize(count);
		}
		if (!empty())
			ar & boost::serialization::make_array(m_storage.data(),size());
		(void) file_version;//prevent warning
	}

private:
	array_type m_storage;
};


template<class T, class Orientation, bool Upper, bool Unit>
class dense_triangular_proxy<T, Orientation, triangular_tag<Upper, Unit> , cpu_tag>
: public matrix_expression<dense_triangular_proxy<T, Orientation, triangular_tag<Upper, Unit>, cpu_tag>, cpu_tag> {
public:
	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type result_type;
	typedef typename std::conditional<Unit, value_type const&, T&>::type reference;
	typedef value_type const& const_reference;
	typedef dense_triangular_proxy<value_type const, Orientation, triangular_tag<Upper, Unit> , cpu_tag> const_closure_type;
	typedef dense_triangular_proxy<T, Orientation, triangular_tag<Upper, Unit> , cpu_tag> closure_type;

	typedef dense_matrix_storage<T, dense_tag> storage_type;
	typedef dense_matrix_storage<value_type const, dense_tag> const_storage_type;

	typedef elementwise<dense_tag> evaluation_category;
	typedef triangular<Orientation,triangular_tag<Upper, Unit> > orientation;


	template<class U>
	dense_triangular_proxy(dense_triangular_proxy<U, Orientation, triangular_tag<Upper, Unit>, cpu_tag> const& expression)
	: m_values(expression.raw_storage().values)
	, m_size1(expression.size1())
	, m_size2(expression.size2())
	, m_leading_dimension(expression.raw_storage().leading_dimension){}

	/// \brief Constructor of a vector proxy from a Dense matrix
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
	dense_triangular_proxy(storage_type const& storage, no_queue, std::size_t size1, std::size_t size2)
	: m_values(storage.values)
	, m_size1(size1)
	, m_size2(size2)
	, m_leading_dimension(storage.leading_dimension){}
	
	dense_matrix_adaptor<T, Orientation, dense_tag, cpu_tag> to_dense() const{
		return {raw_storage(), queue(), m_size1, m_size2};
	}
	
	
	/// \brief Return the number of rows of the matrix
	size_type size1() const {
		return m_size1;
	}
	/// \brief Return the number of columns of the matrix
	size_type size2() const {
		return m_size2;
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage() const{
		return {m_values, m_leading_dimension};
	}
	
	typename device_traits<cpu_tag>::queue_type& queue()const{
		return device_traits<cpu_tag>::default_queue();
	}

	typedef iterators::dense_storage_iterator<value_type> major_iterator;
	typedef iterators::dense_storage_iterator<value_type const> const_major_iterator;
	
	const_major_iterator major_begin(size_type i) const {
		std::size_t start =  Upper? i + Unit: 0;
		return const_major_iterator(m_values + orientation::element(i,start, m_leading_dimension),start, 1);
	}
	const_major_iterator major_end(size_type i) const {
		std::size_t end =  Upper? m_size2: i +1 - Unit;
		return const_major_iterator(m_values + orientation::element(i,end, m_leading_dimension),end, 1);
	}
	major_iterator major_begin(size_type i){
		std::size_t start =  Upper? i + Unit: 0;
		return major_iterator(m_values + orientation::element(i,start, m_leading_dimension),start, 1);
	}
	major_iterator major_end(size_type i){
		std::size_t end =  Upper? m_size2: i + 1 - Unit;
		return major_iterator(m_values + orientation::element(i,end, m_leading_dimension),end, 1);
	}
	
private:
	T* m_values;
	std::size_t m_size1;
	std::size_t m_size2;
	std::size_t m_leading_dimension;
};


namespace detail{
template<class T, class Orientation>
struct vector_to_matrix_optimizer<dense_vector_adaptor<T, continuous_dense_tag, cpu_tag>, Orientation >{
	typedef dense_matrix_adaptor<T, Orientation, continuous_dense_tag, cpu_tag> type;
	
	static type create(
		dense_vector_adaptor<T, continuous_dense_tag, cpu_tag> const& v,
		std::size_t size1, std::size_t size2
	){
		dense_matrix_storage<T, continuous_dense_tag> storage = {v.raw_storage().values, Orientation::index_m(size1,size2)};
		return type(storage, v.queue(), size1, size2);
	}
};


}}

#endif
