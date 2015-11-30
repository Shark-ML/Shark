/*!
 * \brief       Implements a set of matrices
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
#ifndef SHARK_LINALG_BLAS_MATRIX_SET_HPP
#define SHARK_LINALG_BLAS_MATRIX_SET_HPP

#include "matrix_expression.hpp"

#include <boost/serialization/vector.hpp>

namespace shark {
namespace blas {

template<class element_type>
class matrix_set:
	//todo: mpl decision on whether this is a matrix or vector set.
	public matrix_set_expression<matrix_set<element_type> > {
	typedef matrix_set<element_type> self_type;
	typedef std::vector<element_type> array_type;
public:
	typedef typename array_type::size_type size_type;
	typedef typename array_type::difference_type difference_type;
	typedef typename array_type::value_type value_type;
	typedef typename element_type::scalar_type scalar_type;
	typedef typename array_type::const_reference const_reference;
	typedef typename array_type::reference reference;
	typedef value_type const* const_pointer;
	typedef value_type* pointer;

	typedef typename element_type::index_type index_type;
	typedef typename element_type::const_index_pointer const_index_pointer;
	typedef typename element_type::index_pointer index_pointer;

	//~ typedef const matrix_reference<const self_type> const_closure_type;
	//~ typedef matrix_reference<self_type> closure_type;
	typedef typename element_type::storage_category storage_category;
	typedef typename element_type::orientation orientation;

	// Construction and destruction

	/// Default matrix_set constructor. Make a dense matrix_set of size (0,0)
	matrix_set(){}

	/** matrix_set constructor with defined size
	 * \param size number of element matrices
	 */
	matrix_set(size_type size):m_data(size){}

	/** matrix_set constructor with defined size
	 * \param size number of element matrices
	 * \param init initial value for matrices
	 */
	matrix_set(size_type size, value_type const& init):m_data(size, init) {}

	/** Copy-constructor of a dense matrix_set from a matrix_set expression
	 * \param e is a matrix_set expression
	 */
	template<class E>
	matrix_set(matrix_set_expression<E> const& e):m_data(e.size()) {
		assign(e);
	}
	
	matrix_set(size_type size, size_type size1, size_type size2)
	:m_data(size,element_type(size1,size2)){}
	// --------------
	// Sizes
	// --------------
	
	///\brief Returns the number of matrices in the set
	size_type size() const {
		return m_data.size();
	}
	
	///\brief Returns the  first dimension of the matrices in the set
	size_type size1() const {
		return m_data.empty()?0:m_data[0].size1();
	}
	
	///\brief Returns the  second dimension of the matrices in the set
	size_type size2() const {
		return m_data.empty()?0:m_data[0].size2();
	}
	
	// Resizing
	/** Resize a matrix_set to new dimensions. If resizing is performed, the data is not preserved.
	 * \param size the new number of elements
	 */
	void resize(size_type size) {
		m_data.resize(size);
	}

	// --------------
	// Element access
	// --------------

	/// \brief Return a const reference to the element \f$i\f$
	const_reference operator()(index_type i) const {
		return m_data[i];
	}

	/// \brief Return a reference to the element \f$i\f$
	reference operator()(index_type i) {
		return m_data[i];
	}

	/// \brief Return a const reference to the element \f$i\f$
	const_reference operator [](index_type i) const {
		return (*this)(i);
	}

	/// \brief Return a reference to the element \f$i\f$
	reference operator [](index_type i) {
		return (*this)(i);
	}
	
	///\brief Returns the first element of the vector
	reference front(){
		return m_data[0];
	}
	///\brief Returns the first element of the vector
	const_reference front()const{
		return m_data[0];
	}
	///\brief Returns the last element of the vector
	reference back(){
		return m_data[size()-1];
	}
	///\brief Returns the last element of the vector
	const_reference back()const{
		return m_data[size()-1];
	}
	
	// Iterators
	
	// Iterator types
	typedef typename array_type::iterator iterator;
	typedef typename array_type::const_iterator const_iterator;
	
	/// \brief return a const iterator on the first element of the vector
	const_iterator cbegin() const {
		return m_data.begin();
	}
	
	/// \brief return a const iterator after the last element of the vector
	const_iterator cend() const {
		return m_data.end();
	}

	/// \brief return an iterator on the first element of the vector
	const_iterator begin() const {
		return m_data.begin();
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator end() const {
		return m_data.end();
	}

	/// \brief Return an iterator on the first element of the vector
	iterator begin(){
		return m_data.begin();
	}

	/// \brief Return an iterator at the end of the vector
	iterator end(){
		return m_data.end();
	}
	
	// Assignment
	
	template<class E>
	matrix_set& assign(matrix_set_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		for(std::size_t i = 0; i != size(); ++i){
			m_data[i].assign(e[i]);
		}
		return *this;
	}
	template<class E>
	matrix_set& plus_assign(matrix_set_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		for(std::size_t i = 0; i != size(); ++i){
			m_data[i].plus_assign(e[i]);
		}
		return *this;
	}
	template<class E>
	matrix_set& minus_assign(matrix_set_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		for(std::size_t i = 0; i != size(); ++i){
			m_data[i].minus_assign(e[i]);
		}
		return *this;
	}
	
	template<class E>
	matrix_set& multiply_assign(matrix_set_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		for(std::size_t i = 0; i != size(); ++i){
			m_data[i].multiply_assign(e[i]);
		}
		return *this;
	}
	template<class E>
	matrix_set& divide_assign(matrix_set_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		for(std::size_t i = 0; i != size(); ++i){
			m_data[i].divide_assign(e[i]);
		}
		return *this;
	}
	
	/*! @note "pass by value" the key idea to enable move semantics */
	matrix_set& operator = (matrix_set m) {
		swap(m);
		return *this;
	}
	template<class E>
	matrix_set& operator = (matrix_set_expression<E> const& e) {
		self_type temporary(e);
		swap(temporary);
		return *this;
	}
	template<class E>
	matrix_set& operator += (matrix_set_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		for(std::size_t i = 0; i != size(); ++i){
			m_data[i] += e[i];
		}
		return *this;
	}
	
	template<class E>
	matrix_set& operator -= (matrix_set_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		for(std::size_t i = 0; i != size(); ++i){
			m_data[i] -= e[i];
		}
		return *this;
	}
	
	template<class E>
	matrix_set& operator *= (matrix_set_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		for(std::size_t i = 0; i != size(); ++i){
			m_data[i] *= e[i];
		}
		return *this;
	}
	
	template<class E>
	matrix_set& operator /= (matrix_set_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		for(std::size_t i = 0; i != size(); ++i){
			m_data[i] /= e[i];
		}
		return *this;
	}
	
	matrix_set& operator *= (scalar_type t) {
		for(std::size_t i = 0; i != size(); ++i){
			m_data[i] *= t;
		}
		return *this;
	}
	matrix_set& operator /= (scalar_type t) {
		for(std::size_t i = 0; i != size(); ++i){
			m_data[i] /= t;
		}
		return *this;
	}

	// Swapping
	void swap(matrix_set& m) {
		using std::swap;
		swap(m_data, m.m_data);
	}
	
	void clear(){
		for(std::size_t i = 0; i != size(); ++i){
			m_data[i].clear();
		}
	}


	// Serialization
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */) {
		ar& boost::serialization::make_nvp("data",m_data);
	}

private:
	array_type m_data;
};
}}

#endif
