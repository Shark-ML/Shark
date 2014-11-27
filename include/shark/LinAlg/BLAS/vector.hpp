#ifndef SHARK_LINALG_BLAS_VECTOR_HPP
#define SHARK_LINALG_BLAS_VECTOR_HPP

#include "vector_proxy.hpp"
#include <boost/container/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/collection_size_type.hpp>

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
template<class T>
class vector:
	public vector_container<vector<T> > {

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

	// Construction and destruction

	/// \brief Constructor of a vector
	/// By default it is empty, i.e. \c size()==0.
	vector():m_storage() {}

	/// \brief Constructor of a vector with a predefined size
	/// By default, its elements are initialized to 0.
	/// \param size initial size of the vector
	explicit vector(size_type size):m_storage(size) {}
		
	/// \brief Constructs the vector from a predefined range
	template<class Iter>
	vector(Iter begin, Iter end):m_storage(begin,end){}

	/// \todo remove this definition because \c size is not used
	/// \brief Constructor of a vector by copying from another container
	/// This type has the generic name \c array_typ within the vector definition.
	/// \param size initial size of the vector \bug this value is not used
	/// \param data container of type \c A
	vector(size_type size, const array_type& data):m_storage(data) {}

	/// \brief Constructor of a vector by copying from another container
	/// This type has the generic name \c array_typ within the vector definition.
	/// \param data container of type \c A
	vector(const array_type& data):m_storage(data) {}

	/// \brief Constructor of a vector with a predefined size and a unique initial value
	/// \param size of the vector
	/// \param init value to assign to each element of the vector
	vector(size_type size, const value_type& init):m_storage(size, init) {}

	/// \brief Copy-constructor of a vector
	/// \param v is the vector to be duplicated
	vector(const vector& v):m_storage(v.m_storage) {}

	/// \brief Copy-constructor of a vector from a vector_expression
	/// Depending on the vector_expression, this constructor can have the cost of the computations
	/// of the expression (trivial to say it, but it is to take into account in your complexity calculations).
	/// \param e the vector_expression which values will be duplicated into the vector
	template<class E>
	vector(vector_expression<E> const& e):
		vector_container<self_type> (),
		m_storage(e().size()) {
		kernels::assign (*this, e);
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
		return &m_storage[0];
	}
	
	///\brief Returns the pointer to the beginning of the vector storage
	///
	/// Grants low-level access to the vectors internals. Elements storage()[0]...storage()[size()-1] are valid.
	const_pointer storage()const{
		return &m_storage[0];
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
		return m_storage.size() == 0;
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
		return storage()[i];
	}

	/// \brief Return a reference to the element \f$i\f$
	/// Return a reference to the element \f$i\f$. With some compilers, this notation will be faster than \c operator[]
	/// \param i index of the element
	reference operator()(index_type i) {
		return storage()[i];
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator [](index_type i) const {
		return (*this)(i);
	}

	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator [](index_type i) {
		return (*this)(i);
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
	
	// -------------------
	// Assignment Functions
	// -------------------

	/// \brief Assign the result of a vector_expression to the vector
	/// Assign the result of a vector_expression to the vector.
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vector& assign(vector_expression<E> const& e) {
		kernels::assign (*this, e);
		return *this;
	}
	
	/// \brief Assign the sum of the vector and a vector_expression to the vector
	/// Assign the sum of the vector and a vector_expression to the vector.
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vector& plus_assign(vector_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		kernels::assign<scalar_plus_assign> (*this, e);
		return *this;
	}
	
	/// \brief Assign the difference of the vector and a vector_expression to the vector
	/// Assign the difference of the vector and a vector_expression to the vector.
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vector& minus_assign(vector_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		kernels::assign<scalar_minus_assign> (*this, e);
		return *this;
	}
	
	/// \brief Assign the elementwise product of the vector and a vector_expression to the vector
	/// Assign the difference of the vector and a vector_expression to the vector.
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vector& multiply_assign(vector_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		kernels::assign<scalar_multiply_assign> (*this, e);
		return *this;
	}
	
	/// \brief Assign the elementwise division of the vector and a vector_expression to the vector
	/// Assign the difference of the vector and a vector_expression to the vector.
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vector& divide_assign(vector_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		kernels::assign<scalar_divide_assign> (*this, e);
		return *this;
	}

	// -------------------
	// Assignment operators
	// -------------------
	
	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector). This method does not create any temporary.
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	template<class C>          // Container assignment without temporary
	vector& operator = (vector_container<C> const& v) {
		resize(v().size());
		return assign(v);
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
	
	// Assignment
	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// \param v is the source vector
	/// \return a reference to a vector (i.e. the destination vector)
	vector& operator = (vector v) {
		swap(*this,v);
		return *this;
	}
	

	/// \brief Assign the sum of the vector and a vector_expression to the vector
	/// Assign the sum of the vector and a vector_expression to the vector.
	/// A temporary is created for the computations.
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vector& operator += (vector_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		self_type temporary(e);
		return plus_assign(temporary);
	}

	/// \brief Assign the sum of the vector and a vector_expression to the vector
	/// Assign the sum of the vector and a vector_expression to the vector.
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \param v is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class C>          // Container assignment without temporary
	vector& operator += (vector_container<C> const& v) {
		SIZE_CHECK(size() == v().size());
		return plus_assign(v);
	}

	/// \brief Assign the difference of the vector and a vector_expression to the vector
	/// Assign the difference of the vector and a vector_expression to the vector.
	/// A temporary is created for the computations.
	/// \param e is a const reference to the vector_expression
	template<class E>
	vector& operator -= (vector_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		self_type temporary(e);
		return minus_assign(temporary);
	}

	/// \brief Assign the difference of the vector and a vector_expression to the vector
	/// Assign the difference of the vector and a vector_expression to the vector.
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \param v is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class C>          // Container assignment without temporary
	vector& operator -= (vector_container<C> const& v) {
		SIZE_CHECK(size() == v().size());
		return minus_assign(v);
	}
	
	/// \brief Assign the elementwise product of the vector and a vector_expression to the vector
	/// A temporary is created for the computations.
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vector& operator *= (vector_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		self_type temporary(e);
		return multiply_assign(temporary);
	}

	/// \brief Assign the elementwise product of the vector and a vector_expression to the vector
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \param v is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class C>          // Container assignment without temporary
	vector& operator *= (vector_container<C> const& v) {
		SIZE_CHECK(size() == v().size());
		return multiply_assign(v);
	}
	
	/// \brief Assign the elementwise division of the vector and a vector_expression to the vector
	/// A temporary is created for the computations.
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vector& operator /= (vector_expression<E> const& e) {
		SIZE_CHECK(size() == e().size());
		self_type temporary(e);
		return divide_assign(temporary);
	}

	/// \brief Assign the elementwise product of the vector and a vector_expression to the vector
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \param v is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class C>          // Container assignment without temporary
	vector& operator /= (vector_container<C> const& v) {
		SIZE_CHECK(size() == v().size());
		return divide_assign(v);
	}

	/// \brief Assign the product of the vector and a scalar to the vector
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \param t is a const reference to the scalar
	/// \return a reference to the resulting vector
	vector& operator *= (scalar_type t) {
		kernels::assign<scalar_multiply_assign> (*this, t);
		return *this;
	}

	/// \brief Assign the division of the vector by a scalar to the vector
	/// Assign the division of the vector by a scalar to the vector.
	/// No temporary is created. Computations are done and stored directly into the resulting vector.
	/// \tparam E is the type of the vector_expression
	/// \param t is a const reference to the scalar
	/// \return a reference to the resulting vector
	vector& operator /= (scalar_type t) {
		kernels::assign<scalar_divide_assign> (*this, t);
		return *this;
	}

	// Iterator types
	typedef dense_storage_iterator<value_type> iterator;
	typedef dense_storage_iterator<value_type const> const_iterator;
	
	/// \brief return an iterator on the first element of the vector
	const_iterator cbegin() const {
		return const_iterator(&m_storage[0],0);
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator cend() const {
		return const_iterator(&m_storage[0]+size(),size());
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
		return iterator(&m_storage[0],0);
	}

	/// \brief Return an iterator at the end of the vector
	iterator end(){
		return iterator(&m_storage[0]+size(),size());
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

}
}

#endif
