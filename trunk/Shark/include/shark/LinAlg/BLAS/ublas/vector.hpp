//
//  Copyright (c) 2000-2010
//  Joerg Walter, Mathias Koch, David Bellot
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  GeNeSys mbH & Co. KG in producing this work.
//
//  And we acknowledge the support from all contributors.

/// \file vector.hpp Definition for the class vector and its derivative

#ifndef _BOOST_UBLAS_VECTOR_
#define _BOOST_UBLAS_VECTOR_

#include <shark/LinAlg/BLAS/ublas/storage.hpp>
#include <shark/LinAlg/BLAS/ublas/vector_expression.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/vector_assign.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/nvp.hpp>


// Iterators based on ideas of Jeremy Siek

namespace shark {
namespace blas {

/** \brief A dense vector of values of type \c T.
 *
 * For a \f$n\f$-dimensional vector \f$v\f$ and \f$0\leq i < n\f$ every element \f$v_i\f$ is mapped
 * to the \f$i\f$-th element of the container. A storage type \c A can be specified which defaults to \c unbounded_array.
 * Elements are constructed by \c A, which need not initialise their value.
 *
 * \tparam T type of the objects stored in the vector (like int, double, complex,...)
 * \tparam A The type of the storage array of the vector. Default is \c unbounded_array<T>. \c <bounded_array<T> and \c std::vector<T> can also be used
 */
template<class T, class A>
class vector:
	public vector_container<vector<T, A> > {

	typedef vector<T, A> self_type;
public:
	typedef typename A::size_type size_type;
	typedef typename A::difference_type difference_type;
	typedef T value_type;
	typedef T const& const_reference;
	typedef T &reference;
	typedef T *pointer;
	typedef const T *const_pointer;
	typedef A array_type;
	typedef const vector_reference<const self_type> const_closure_type;
	typedef vector_reference<self_type> closure_type;
	typedef self_type vector_temporary_type;
	typedef dense_tag storage_category;

	// Construction and destruction

	/// \brief Constructor of a vector
	/// By default it is empty, i.e. \c size()==0.
	vector():
		vector_container<self_type> (),
		data_() {}

	/// \brief Constructor of a vector with a predefined size
	/// By default, its elements are initialized to 0.
	/// \param size initial size of the vector
	explicit vector(size_type size):
		vector_container<self_type> (),
		data_(size) {
	}

	/// \brief Constructor of a vector by copying from another container
	/// This type has the generic name \c array_typ within the vector definition.
	/// \param size initial size of the vector \bug this value is not used
	/// \param data container of type \c A
	/// \todo remove this definition because \c size is not used
	vector(size_type size, const array_type &data):
		vector_container<self_type> (),
		data_(data) {}

	/// \brief Constructor of a vector by copying from another container
	/// This type has the generic name \c array_typ within the vector definition.
	/// \param data container of type \c A
	vector(const array_type &data):
		vector_container<self_type> (),
		data_(data) {}

	/// \brief Constructor of a vector with a predefined size and a unique initial value
	/// \param size of the vector
	/// \param init value to assign to each element of the vector
	vector(size_type size, const value_type &init):
		vector_container<self_type> (),
		data_(size, init) {}

	/// \brief Copy-constructor of a vector
	/// \param v is the vector to be duplicated
	vector(const vector &v):
		vector_container<self_type> (),
		data_(v.data_) {}

	/// \brief Copy-constructor of a vector from a vector_expression
	/// Depending on the vector_expression, this constructor can have the cost of the computations
	/// of the expression (trivial to say it, but it is to take into account in your complexity calculations).
	/// \param ae the vector_expression which values will be duplicated into the vector
	template<class AE>
	vector(const vector_expression<AE> &ae):
		vector_container<self_type> (),
		data_(ae().size()) {
		vector_assign<scalar_assign> (*this, ae);
	}

	// -----------------------
	// Random Access Container
	// -----------------------

	/// \brief Return the maximum size of the data container.
	/// Return the upper bound (maximum size) on the data container. Depending on the container, it can be bigger than the current size of the vector.
	size_type max_size() const {
		return data_.max_size();
	}

	/// \brief Return true if the vector is empty (\c size==0)
	/// \return \c true if empty, \c false otherwise
	bool empty() const {
		return data_.size() == 0;
	}

	// ---------
	// Accessors
	// ---------

	/// \brief Return the size of the vector
	size_type size() const {
		return data_.size();
	}

	// -----------------
	// Storage accessors
	// -----------------

	/// \brief Return a \c const reference to the container. Useful to access data directly for specific type of container.
	const array_type &data() const {
		return data_;
	}

	/// \brief Return a reference to the container. Useful to speed-up write operations to the data in very specific case.
	array_type &data() {
		return data_;
	}

	// --------
	// Resizing
	// --------

	/// \brief Resize the vector
	/// Resize the vector to a new size. If \c preserve is true, data are copied otherwise data are lost. If the new size is bigger, the remaining values are filled in with the initial value (0 by default) in the case of \c unbounded_array, which is the container by default. If the new size is smaller, last values are lost. This behaviour can be different if you explicitely specify another type of container.
	/// \param size new size of the vector
	/// \param preserve if true, keep values
	void resize(size_type size, bool preserve = true) {
		if (preserve)
			data().resize(size, typename A::value_type());
		else
			data().resize(size);
	}

	// ---------------
	// Element support
	// ---------------

	/// \brief Return a pointer to the element \f$i\f$
	/// \param i index of the element
	// XXX this semantic is not the one expected by the name of this method
	pointer find_element(size_type i) {
		return const_cast<pointer>(const_cast<const self_type &>(*this).find_element(i));
	}

	/// \brief Return a const pointer to the element \f$i\f$
	/// \param i index of the element
	// XXX  this semantic is not the one expected by the name of this method
	const_pointer find_element(size_type i) const {
		return & (data() [i]);
	}

	// --------------
	// Element access
	// --------------

	/// \brief Return a const reference to the element \f$i\f$
	/// Return a const reference to the element \f$i\f$. With some compilers, this notation will be faster than \c[ i ]
	/// \param i index of the element
	const_reference operator()(size_type i) const {
		return data() [i];
	}

	/// \brief Return a reference to the element \f$i\f$
	/// Return a reference to the element \f$i\f$. With some compilers, this notation will be faster than \c[ i ]
	/// \param i index of the element
	reference operator()(size_type i) {
		return data() [i];
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator [](size_type i) const {
		return (*this)(i);
	}

	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator [](size_type i) {
		return (*this)(i);
	}

	// ------------------
	// Element assignment
	// ------------------

	/// \brief Set element \f$i\f$ to the value \c t
	/// \param i index of the element
	/// \param t reference to the value to be set
	// XXX semantic of this is to insert a new element and therefore size=size+1 ?
	
	reference insert_element(size_type i, const_reference t) {
		return (data() [i] = t);
	}

	/// \brief Set element \f$i\f$ to the \e zero value
	/// \param i index of the element
	void erase_element(size_type i) {
		data() [i] = value_type/*zero*/();
	}

	// -------
	// Zeroing
	// -------

	/// \brief Clear the vector, i.e. set all values to the \c zero value.
	void clear() {
		std::fill(data().begin(), data().end(), value_type/*zero*/());
	}

	// Assignment
	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// \param v is the source vector
	/// \return a reference to a vector (i.e. the destination vector)
	vector &operator = (vector v) {
		assign_temporary(v);
		return *this;
	}

	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector). This method does not create any temporary.
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	template<class C>          // Container assignment without temporary
	vector &operator = (const vector_container<C> &v) {
		resize(v().size(), false);
		assign(v);
		return *this;
	}

	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// \param v is the source vector
	/// \return a reference to a vector (i.e. the destination vector)
	vector &assign_temporary(vector &v) {
		swap(v);
		return *this;
	}

	/// \brief Assign the result of a vector_expression to the vector
	/// Assign the result of a vector_expression to the vector. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// \tparam AE is the type of the vector_expression
	/// \param ae is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class AE>
	vector &operator = (const vector_expression<AE> &ae) {
		self_type temporary(ae);
		return assign_temporary(temporary);
	}

	/// \brief Assign the result of a vector_expression to the vector
	/// Assign the result of a vector_expression to the vector. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// \tparam AE is the type of the vector_expression
	/// \param ae is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class AE>
	vector &assign(const vector_expression<AE> &ae) {
		vector_assign<scalar_assign> (*this, ae);
		return *this;
	}

	// -------------------
	// Computed assignment
	// -------------------

	/// \brief Assign the sum of the vector and a vector_expression to the vector
	/// Assign the sum of the vector and a vector_expression to the vector. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// A temporary is created for the computations.
	/// \tparam AE is the type of the vector_expression
	/// \param ae is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class AE>
	vector &operator += (const vector_expression<AE> &ae) {
		self_type temporary(*this + ae);
		return assign_temporary(temporary);
	}

	/// \brief Assign the sum of the vector and a vector_expression to the vector
	/// Assign the sum of the vector and a vector_expression to the vector. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \tparam C is the type of the vector_expression
	/// \param v is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class C>          // Container assignment without temporary
	vector &operator += (const vector_container<C> &v) {
		plus_assign(v);
		return *this;
	}

	/// \brief Assign the sum of the vector and a vector_expression to the vector
	/// Assign the sum of the vector and a vector_expression to the vector. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \tparam AE is the type of the vector_expression
	/// \param ae is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class AE>
	vector &plus_assign(const vector_expression<AE> &ae) {
		vector_assign<scalar_plus_assign> (*this, ae);
		return *this;
	}

	/// \brief Assign the difference of the vector and a vector_expression to the vector
	/// Assign the difference of the vector and a vector_expression to the vector. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// A temporary is created for the computations.
	/// \tparam AE is the type of the vector_expression
	/// \param ae is a const reference to the vector_expression
	template<class AE>
	vector &operator -= (const vector_expression<AE> &ae) {
		self_type temporary(*this - ae);
		return assign_temporary(temporary);
	}

	/// \brief Assign the difference of the vector and a vector_expression to the vector
	/// Assign the difference of the vector and a vector_expression to the vector. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \tparam C is the type of the vector_expression
	/// \param v is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class C>          // Container assignment without temporary
	vector &operator -= (const vector_container<C> &v) {
		minus_assign(v);
		return *this;
	}

	/// \brief Assign the difference of the vector and a vector_expression to the vector
	/// Assign the difference of the vector and a vector_expression to the vector. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \tparam AE is the type of the vector_expression
	/// \param ae is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class AE>
	vector &minus_assign(const vector_expression<AE> &ae) {
		vector_assign<scalar_minus_assign> (*this, ae);
		return *this;
	}

	/// \brief Assign the product of the vector and a scalar to the vector
	/// Assign the product of the vector and a scalar to the vector. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \tparam AE is the type of the vector_expression
	/// \param at is a const reference to the scalar
	/// \return a reference to the resulting vector
	template<class AT>
	vector &operator *= (const AT &at) {
		vector_assign_scalar<scalar_multiplies_assign> (*this, at);
		return *this;
	}

	/// \brief Assign the division of the vector by a scalar to the vector
	/// Assign the division of the vector by a scalar to the vector. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \tparam AE is the type of the vector_expression
	/// \param at is a const reference to the scalar
	/// \return a reference to the resulting vector
	template<class AT>
	vector &operator /= (const AT &at) {
		vector_assign_scalar<scalar_divides_assign> (*this, at);
		return *this;
	}

	// --------
	// Swapping
	// --------

	/// \brief Swap the content of the vector with another vector
	/// \param v is the vector to be swapped with
	void swap(vector &v) {
		if (this != &v) {
			data().swap(v.data());
		}
	}

	/// \brief Swap the content of two vectors
	/// \param v1 is the first vector. It takes values from v2
	/// \param v2 is the second vector It takes values from v1
	friend void swap(vector &v1, vector &v2) {
		v1.swap(v2);
	}

	// Iterator types
private:
	// Use the storage array iterator
	typedef typename A::const_iterator const_subiterator_type;
	typedef typename A::iterator subiterator_type;

public:
	typedef indexed_iterator<self_type, dense_random_access_iterator_tag> iterator;
	typedef indexed_const_iterator<self_type, dense_random_access_iterator_tag> const_iterator;

	// --------------
	// Element lookup
	// --------------

	/// \brief Return a const iterator to the element \e i
	/// \param i index of the element
	
	const_iterator find(size_type i) const {
		return const_iterator(*this, i);
	}

	/// \brief Return an iterator to the element \e i
	/// \param i index of the element
	iterator find(size_type i) {
		return iterator(*this, i);
	}

	/// \brief return an iterator on the first element of the vector
	const_iterator begin() const {
		return find(0);
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator end() const {
		return find(data_.size());
	}

	/// \brief Return an iterator on the first element of the vector
	iterator begin() {
		return find(0);
	}

	/// \brief Return an iterator at the end of the vector
	iterator end() {
		return find(data_.size());
	}

	// Reverse iterator
	typedef reverse_iterator_base<const_iterator> const_reverse_iterator;
	typedef reverse_iterator_base<iterator> reverse_iterator;

	/// \brief Return a const reverse iterator before the first element of the reversed vector (i.e. end() of normal vector)
	const_reverse_iterator rbegin() const {
		return const_reverse_iterator(end());
	}

	/// \brief Return a const reverse iterator on the end of the reverse vector (i.e. first element of the normal vector)
	const_reverse_iterator rend() const {
		return const_reverse_iterator(begin());
	}

	/// \brief Return a const reverse iterator before the first element of the reversed vector (i.e. end() of normal vector)
	reverse_iterator rbegin() {
		return reverse_iterator(end());
	}

	/// \brief Return a const reverse iterator on the end of the reverse vector (i.e. first element of the normal vector)
	reverse_iterator rend() {
		return reverse_iterator(begin());
	}

	// -------------
	// Serialization
	// -------------

	/// Serialize a vector into and archive as defined in Boost
	/// \param ar Archive object. Can be a flat file, an XML file or any other stream
	/// \param file_version Optional file version (not yet used)
	template<class Archive>
	void serialize(Archive &ar, const unsigned int /* file_version */) {
		ar &boost::serialization::make_nvp("data",data_);
	}

private:
	array_type data_;
};

/// \brief A scalar (i.e. unique value) vector of type \c T and a given \c size
/// A scalar (i.e. unique value) vector of type \c T and a given \c size. This is a virtual vector in the sense that no memory is allocated
/// for storing the unique value more than once: it still acts like any other vector. However assigning a new value will change all the value at once.
/// vector into a normal vector. It must first be assigned to another normal vector by any suitable means. Its memory footprint is constant.
/// \tparam T type of the objects stored in the vector: it can be anything even if most of the time, scalar types will be used like \c double or \c int. Coboost::mplex types can be used, or even classes like boost::interval.
template<class T>
class scalar_vector:
	public vector_container<scalar_vector<T> > {

	typedef const T *const_pointer;
	typedef scalar_vector<T> self_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef const T &const_reference;
	typedef T &reference;
	typedef const vector_reference<const self_type> const_closure_type;
	typedef vector_reference<self_type> closure_type;
	typedef dense_tag storage_category;

	// Construction and destruction
	scalar_vector()
	:size_(0), value_() {}
	explicit scalar_vector(size_type size, value_type value)
	:size_(size), value_(value) {}
	scalar_vector(const scalar_vector &v)
	:size_(v.size_), value_(v.value_) {}

	// Accessors
	size_type size() const {
		return size_;
	}

	// Resizing
	void resize(size_type size, bool /*preserve*/ = true) {
		size_ = size;
	}

	// Element support
	const_pointer find_element(size_type /*i*/) const {
		return & value_;
	}

	// Element access
	const_reference operator()(size_type /*i*/) const {
		return value_;
	}

	const_reference operator [](size_type /*i*/) const {
		return value_;
	}

public:
	typedef indexed_const_iterator<self_type, dense_random_access_iterator_tag> iterator;
	typedef indexed_const_iterator<self_type, dense_random_access_iterator_tag> const_iterator;

	// Element lookup
	const_iterator find(size_type i) const {
		return const_iterator(*this, i);
	}
	const_iterator begin() const {
		return find(0);
	}
	const_iterator end() const {
		return find(size_);
	}

	// Reverse iterator
	typedef reverse_iterator_base<const_iterator> const_reverse_iterator;
	typedef const_reverse_iterator reverse_iterator;

	const_reverse_iterator rbegin() const {
		return const_reverse_iterator(end());
	}
	const_reverse_iterator rend() const {
		return const_reverse_iterator(begin());
	}

private:
	size_type size_;
	value_type value_;
};

// -----------------
// Zero vector class
// -----------------

/// \brief A zero vector of type \c T and a given \c size
/// A zero vector of type \c T and a given \c size.
/// This is a special case of the scalar vector for constant zero value
template<class T>
struct zero_vector: public scalar_vector<T> {
	explicit zero_vector(typename scalar_vector<T>::size_type size)
	:scalar_vector<T>(size,T()){}
};

}
}

#endif
