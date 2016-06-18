/*!
 * \brief       Implements the Dense vector class
 * 
 * \author      O. Krause
 * \date        2014
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
#ifndef SHARK_LINALG_BLAS_VECTOR_HPP
#define SHARK_LINALG_BLAS_VECTOR_HPP

#include "vector_proxy.hpp"
#include <boost/container/vector.hpp>
#include <array>
#include <initializer_list>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/array.hpp>

namespace shark {
namespace blas {

/// \brief A dense vector of values of type \c T.
///
/// For a \f$n\f$-dimensional vector \f$v\f$ and \f$0\leq i < n\f$ every element \f$v_i\f$ is mapped
/// to the \f$i\f$-th element of the container.
///
/// \tparam T type of the objects stored in the vector (like int, double, complex,...)
template<class T>
class vector: public vector_container<vector<T> > {

	typedef vector<T> self_type;
	typedef boost::container::vector<T> array_type;
public:
	
	typedef typename array_type::size_type size_type;
	typedef typename array_type::difference_type difference_type;
	typedef typename array_type::value_type value_type;
	typedef value_type scalar_type;
	typedef typename array_type::const_reference const_reference;
	typedef typename array_type::reference reference;
	typedef T *pointer;
	typedef const T *const_pointer;

	typedef std::size_t index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type index_pointer;

	typedef const vector_reference<const self_type> const_closure_type;
	typedef vector_reference<self_type> closure_type;
	typedef self_type vector_temporary_type;
	typedef dense_tag storage_category;
	typedef elementwise_tag evaluation_category;

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

	/// \brief Copy-constructor of a vector from a vector_expression
	/// \param e the vector_expression whose values will be duplicated into the vector
	template<class E>
	vector(vector_expression<E> const& e):m_storage(e().size()) {
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
	vector& operator = (vector_container<C> const& v) {
		resize(v().size());
		return assign(*this, v);
	}

	/// \brief Assign the result of a vector_expression to the vector
	/// Assign the result of a vector_expression to the vector.
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vector& operator = (vector_expression<E> const& e) {
		self_type temporary(e);
		swap(*this,temporary);
		return *this;
	}

	// ---------
	// Dense low level interface
	// ---------
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_storage.size();
	}
	
	///\brief Returns the pointer to the beginning of the vector storage
	///
	/// Grants low-level access to the vectors internals. Elements storage()[0]...storage()[size()-1] are valid.
	pointer storage(){
		return m_storage.data();
	}
	
	///\brief Returns the pointer to the beginning of the vector storage
	///
	/// Grants low-level access to the vectors internals. Elements storage()[0]...storage()[size()-1] are valid.
	const_pointer storage()const{
		return m_storage.data();
	}
	
	///\brief Returns the stride between the elements in storage()
	///
	/// In general elements of dense storage entities are spaced like storage()[i*stride()] for i=1,...,size()-1
	/// However for vector strid is guaranteed to be 1.
	difference_type stride()const{
		return 1;
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
	const_reference operator()(index_type i) const {
		RANGE_CHECK(i < size());
		return m_storage[i];
	}

	/// \brief Return a reference to the element \f$i\f$
	/// Return a reference to the element \f$i\f$. With some compilers, this notation will be faster than \c operator[]
	/// \param i index of the element
	reference operator()(index_type i) {
		RANGE_CHECK(i < size());
		return m_storage[i];
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator [](index_type i) const {
		RANGE_CHECK(i < size());
		return m_storage[i];
	}

	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator [](index_type i) {
		RANGE_CHECK(i < size());
		return m_storage[i];
	}
	
	///\brief Returns the first element of the vector
	reference front(){
		return storage()[0];
	}
	///\brief Returns the first element of the vector
	const_reference front()const{
		return storage()[0];
	}
	///\brief Returns the last element of the vector
	reference back(){
		return storage()[size()-1];
	}
	///\brief Returns the last element of the vector
	const_reference back()const{
		return storage()[size()-1];
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
	typedef dense_storage_iterator<value_type> iterator;
	typedef dense_storage_iterator<value_type const> const_iterator;
	
	/// \brief return an iterator on the first element of the vector
	const_iterator cbegin() const {
		return const_iterator(storage(),0);
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator cend() const {
		return const_iterator(storage()+size(),size());
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
		return iterator(storage(),0);
	}

	/// \brief Return an iterator at the end of the vector
	iterator end(){
		return iterator(storage()+size(),size());
	}
	
	/////////////////sparse interface///////////////////////////////
	iterator set_element(iterator pos, index_type index, value_type value) {
		SIZE_CHECK(pos.index() == index);
		(*this)(index) = value;
		
		return pos;
	}

	iterator clear_element(iterator pos) {
		SIZE_CHECK(pos != end());
		v(pos.index()) = value_type();
		
		//return new iterator to the next element
		return pos+1;
	}
	
	iterator clear_range(iterator start, iterator end) {
		RANGE_CHECK(start <= end);
		std::fill(start,end,value_type());
		return end;
	}
	
	void reserve(size_type) {}
	
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
			ar & boost::serialization::make_array(storage(),size());
		(void) file_version;//prevent warning
	}

private:
	array_type m_storage;
};

template<class T>
struct vector_temporary_type<T,dense_random_access_iterator_tag>{
	typedef vector<T> type;
};

template<class T>
struct const_expression<vector<T> >{
	typedef vector<T> const type;
};
template<class T>
struct const_expression<vector<T> const>{
	typedef vector<T> const type;
};


/// \brief A dense vector of values of type \c T.
///
/// For a \f$N\f$-dimensional vector \f$v\f$ and \f$0\leq i < N\f$ every element \f$v_i\f$ is mapped
/// to the \f$i\f$-th element of the container.
///
/// \tparam T type of the objects stored in the vector (like int, double, complex,...)
template<class T, std::size_t N>
class vectorN: public vector_container<vectorN<T,N> > {

	typedef vectorN<T,N> self_type;
	typedef std::array<T,N> array_type;
public:
	
	typedef typename array_type::size_type size_type;
	typedef typename array_type::difference_type difference_type;
	typedef typename array_type::value_type value_type;
	typedef value_type scalar_type;
	typedef typename array_type::const_reference const_reference;
	typedef typename array_type::reference reference;
	typedef T *pointer;
	typedef const T *const_pointer;

	typedef std::size_t index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type index_pointer;

	typedef const vector_reference<const self_type> const_closure_type;
	typedef vector_reference<self_type> closure_type;
	typedef self_type vector_temporary_type;
	typedef dense_tag storage_category;
	typedef elementwise_tag evaluation_category;

	// Construction and assignment
	vectorN() = default;
	vectorN(vectorN const& v) = default;
	//~ vectorN(vectorN&& v) = default;
	template<class... Init>
	vectorN(Init&&... init):m_storage({T(init)...}){
		static_assert(sizeof...(Init) == N, "initialisation must have same number of elements as array size");
	}
	/// \brief Copy-constructor of a vector from a vector_expression
	/// \param e the vector_expression which values will be duplicated into the vector. Must have size N.
	template<class E>
	vectorN(vector_expression<E> const& e){
		SIZE_CHECK(e().size() == N);
		assign(*this, e);
	}

	// ---------
	// Dense low level interface
	// ---------
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_storage.size();
	}
	
	///\brief Returns the pointer to the beginning of the vector storage
	///
	/// Grants low-level access to the vectors internals. Elements storage()[0]...storage()[size()-1] are valid.
	pointer storage(){
		return m_storage.data();
	}
	
	///\brief Returns the pointer to the beginning of the vector storage
	///
	/// Grants low-level access to the vectors internals. Elements storage()[0]...storage()[size()-1] are valid.
	const_pointer storage()const{
		return m_storage.data();
	}
	
	///\brief Returns the stride between the elements in storage()
	///
	/// In general elements of dense storage entities are spaced like storage()[i*stride()] for i=1,...,size()-1
	/// However for vector strid is guaranteed to be 1.
	difference_type stride()const{
		return 1;
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

	// --------------
	// Element access
	// --------------

	/// \brief Return a const reference to the element \f$i\f$
	/// Return a const reference to the element \f$i\f$.
	/// \param i index of the element
	const_reference operator()(index_type i) const {
		RANGE_CHECK(i < size());
		return storage()[i];
	}

	/// \brief Return a reference to the element \f$i\f$
	/// Return a reference to the element \f$i\f$.
	/// \param i index of the element
	reference operator()(index_type i) {
		RANGE_CHECK(i < size());
		return storage()[i];
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator [](index_type i) const {
		RANGE_CHECK(i < size());
		return (*this)(i);
	}

	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator [](index_type i) {
		RANGE_CHECK(i < size());
		return (*this)(i);
	}
	
	///\brief Returns the first element of the vector
	reference front(){
		return storage().front();
	}
	///\brief Returns the first element of the vector
	const_reference front()const{
		return storage().front();
	}
	///\brief Returns the last element of the vector
	reference back(){
		return storage().back();
	}
	///\brief Returns the last element of the vector
	const_reference back()const{
		return storage().back();
	}

	/// \brief Clear the vector, i.e. set all values to the \c zero value.
	void clear() {
		std::fill(m_storage.begin(), m_storage.end(), value_type/*zero*/());
	}

	// -------------------
	// Assignment operators
	// -------------------
	
	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector). This method does not create any temporary.
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	vectorN& operator=(vectorN const&) = default;
	
	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector). This method does not create any temporary.
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	template<class C>          // Container assignment without temporary
	vectorN& operator = (vector_container<C> const& v) {
		SIZE_CHECK(v().size() == N);
		resize(v().size());
		return assign(*this, v);
	}

	/// \brief Assign the result of a vector_expression to the vector
	/// Assign the result of a vector_expression to the vector.
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vectorN& operator = (vector_expression<E> const& e) {
		SIZE_CHECK(e().size() == N);
		self_type temporary(e);
		swap(*this,temporary);
		return *this;
	}

	// Iterator types
	typedef dense_storage_iterator<value_type> iterator;
	typedef dense_storage_iterator<value_type const> const_iterator;
	
	/// \brief return an iterator on the first element of the vector
	const_iterator cbegin() const {
		return const_iterator(storage(),0);
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator cend() const {
		return const_iterator(storage()+size(),size());
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
		return iterator(storage(),0);
	}

	/// \brief Return an iterator at the end of the vector
	iterator end(){
		return iterator(storage()+size(),size());
	}
	
	/////////////////sparse interface///////////////////////////////
	iterator set_element(iterator pos, index_type index, value_type value) {
		SIZE_CHECK(pos.index() == index);
		(*this)(index) = value;
		
		return pos;
	}

	iterator clear_element(iterator pos) {
		SIZE_CHECK(pos != end());
		v(pos.index()) = value_type();
		
		//return new iterator to the next element
		return pos+1;
	}
	
	iterator clear_range(iterator start, iterator end) {
		RANGE_CHECK(start <= end);
		std::fill(start,end,value_type());
		return end;
	}
	
	void reserve(size_type) {}
	
	/// \brief Swap the content of two vectors
	/// \param v1 is the first vector. It takes values from v2
	/// \param v2 is the second vector It takes values from v1
	friend void swap(vectorN& v1, vectorN& v2) {
		v1.m_storage.swap(v2.m_storage);
	}
	// -------------
	// Serialization
	// -------------

	template<class Archive>
	void serialize(Archive &ar, const unsigned int) {
		ar & boost::serialization::make_array(storage(),size());
	}

private:
	array_type m_storage;
};

template<class T, std::size_t N>
struct const_expression<vectorN<T,N> >{
	typedef vectorN<T,N> const type;
};
template<class T, std::size_t N>
struct const_expression<vectorN<T,N> const>{
	typedef vectorN<T,N> const type;
};


}
}

#endif
