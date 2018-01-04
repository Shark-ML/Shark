/*!
 * \brief       Implements the dense vector class for the cpu
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
#ifndef REMORA_CPU_VECTOR_HPP
#define REMORA_CPU_VECTOR_HPP

//INCLUDED BY ../vector.hpp!

#include "../detail/vector_proxy_classes.hpp"
#include <initializer_list>
#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/array.hpp>

namespace remora{
template<class T>
class vector<T,cpu_tag>: public vector_container<vector<T, cpu_tag>, cpu_tag > {

	typedef vector<T> self_type;
	typedef std::vector<typename std::conditional<std::is_same<T,bool>::value,char,T>::type > array_type;
public:
	typedef typename array_type::value_type value_type;
	typedef typename array_type::const_reference const_reference;
	typedef typename array_type::reference reference;
	typedef typename array_type::size_type size_type;

	typedef vector_reference<self_type const> const_closure_type;
	typedef vector_reference<self_type> closure_type;
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
		self_type temporary(e);
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
	typename device_traits<cpu_tag>::queue_type& queue(){
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
	
	/////////////////sparse interface///////////////////////////////
	iterator set_element(iterator pos, size_type index, value_type value) {
		REMORA_SIZE_CHECK(pos.index() == index);
		(*this)(index) = value;
		
		return pos;
	}

	iterator clear_element(iterator pos) {
		REMORA_SIZE_CHECK(pos != end());
		v(pos.index()) = value_type();
		
		//return new iterator to the next element
		return pos+1;
	}
	
	iterator clear_range(iterator start, iterator end) {
		REMORA_RANGE_CHECK(start <= end);
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
			ar & boost::serialization::make_array(m_storage.data(),size());
		(void) file_version;//prevent warning
	}

private:
	array_type m_storage;
};

}

#endif
