/*!
 * 
 *
 * \brief       Vector proxy classes.
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


#ifndef SHARK_LINALG_BLAS_VECTOR_PROXY_HPP
#define SHARK_LINALG_BLAS_VECTOR_PROXY_HPP

#include "assignment.hpp"
#include "detail/iterator.hpp"

namespace shark{
namespace blas{
	
template<class V>
class vector_reference:public vector_expression<vector_reference<V> >{
public:
	typedef typename V::size_type size_type;
	typedef typename V::difference_type difference_type;
	typedef typename V::value_type value_type;
	typedef typename V::scalar_type scalar_type;
	typedef typename V::const_reference const_reference;
	typedef typename reference<V>::type reference;
	typedef typename V::const_pointer const_pointer;
	typedef typename pointer<V>::type pointer;

	typedef typename V::index_type index_type;
	typedef typename V::const_index_pointer const_index_pointer;
	typedef typename index_pointer<V>::type index_pointer;
	
	typedef vector_reference<V const> const_closure_type;
	typedef vector_reference<V> closure_type;
	typedef typename V::storage_category storage_category;
	typedef elementwise_tag evaluation_category;

	// Construction
	vector_reference(V& v):m_expression(&v){}
		
	template<class E>
	vector_reference(vector_reference<E> const& other)
		:m_expression(&other.expression()){}
		
	// Expression accessors
	V& expression() const{
		return *m_expression;
	}
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return expression().size();
	}
	
	void clear(){
		expression().clear();
	}

	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the pointer to the beginning of the vector storage
	///
	/// Low-level access to the vectors internals. Elements storage()[i*stride()] for i=1,...,size()-1 are valid
	pointer storage()const{
		return expression().storage();
	}
	
	///\brief Returns th stride between the elements in storage()
	///
	/// In general elements of dense storage entities are spaced like storage()[i*stride()] for i=1,...,size()-1
	difference_type stride()const{
		return expression().stride();
	}
	
	// ---------
	// Sparse low level interface
	// ---------
	
	/// \brief Number of nonzero elements of the vector.
	size_type nnz()const{
		return expression().nnz();
	}
	/// \brief Array of values of the nonzero elements.
	const_pointer values()const{
		return expression().values();
	}
	/// \brief Array of indices of the nonzero elements.
	index_pointer indices()const{
		return expression().indices();
	}
	
	// ---------
	// High level interface
	// ---------
	
	// Element access
	reference operator()(index_type i) const{
		return expression()(i);
	}
	reference operator [](index_type i) const{
		return expression() [i];
	}
	
	vector_reference& operator = (vector_reference const& v){
		expression() = v.expression();
		return *this;
	}
	template<class E>
	vector_reference& operator = (vector_expression<E> const& e){
		expression() = e();
		return *this;
	}

	// Iterator types
	typedef typename V::const_iterator const_iterator;
	typedef typename boost::mpl::if_<boost::is_const<V>,
	        typename V::const_iterator,
	        typename V::iterator>::type iterator;

	// Iterator is the iterator of the referenced expression.
	const_iterator begin() const{
		return expression().begin();
	}
	const_iterator end() const{
		return expression().end();
	}
	iterator begin(){
		return expression().begin();
	}
	iterator end(){
		return expression().end();
	}
	
	//sparse interface
	iterator set_element(iterator pos, index_type index, value_type value){
		return expression().set_element(pos,index,value);
	}

	iterator clear_element(iterator pos){
		return expression().clear_element(pos);
	}
	iterator clear_range(iterator start, iterator end){
		return expression().clear_range(start,end);
	}
	
	void reserve(size_type non_zeros){
		expression().reserve(non_zeros);
	}

private:
	V* m_expression;
};

/** \brief A vector referencing a continuous subvector of elements of vector \c v containing all elements specified by \c range.
 *
 * A vector range can be used as a normal vector in any expression.
 * If the specified range falls outside that of the index range of the vector, then
 * the \c vector_range is not a well formed \c vector_expression and access to an
 * element outside of index range of the vector is \b undefined.
 *
 * \tparam V the type of vector referenced (for exaboost::mple \c vector<double>)
 */
template<class V>
class vector_range:public vector_expression<vector_range<V> >{
public:
	typedef typename V::size_type size_type;
	typedef typename V::difference_type difference_type;
	typedef typename V::value_type value_type;
	typedef typename V::scalar_type scalar_type;
	typedef typename V::const_reference const_reference;
	typedef typename reference<V>::type reference;
	typedef typename V::const_pointer const_pointer;
	typedef typename pointer<V>::type pointer;

	typedef typename V::index_type index_type;
	typedef typename V::const_index_pointer const_index_pointer;
	typedef typename index_pointer<V>::type index_pointer;

	typedef typename closure<V>::type vector_closure_type;
	typedef vector_range<typename const_expression<V>::type> const_closure_type;
	typedef vector_range<V> closure_type;
	typedef typename V::storage_category storage_category;
	typedef elementwise_tag evaluation_category;

	// Construction and destruction
	vector_range(vector_closure_type const& data, range const& r):
		m_expression(data), m_range(r){
		RANGE_CHECK(start() <= m_expression.size());
		RANGE_CHECK(start() + size() <= m_expression.size());
	}
	
	//non-const-> const conversion
	template<class E>
	vector_range(
		vector_range<E> const& other,
		typename boost::disable_if<
			boost::is_same<E,vector_range>
		>::type* dummy = 0
	):m_expression(other.expression())
	, m_range(other.range()){}
	
	// ---------
	// Internal Accessors
	// ---------
	
	size_type start() const{
		return m_range.start();
	}
	
	vector_closure_type const& expression() const{
		return m_expression;
	}
	vector_closure_type& expression(){
		return m_expression;
	}
	
	blas::range const& range()const{
		return m_range;
	}
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_range.size();
	}

	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the pointer to the beginning of the vector storage
	///
	/// Low-level access to the vectors internals. Elements storage()[i*stride()] for i=1,...,size()-1 are valid
	pointer storage()const{
		return expression().storage()+start()*stride();
	}
	
	///\brief Returns the stride between the elements in storage()
	///
	/// In general elements of dense storage entities are spaced like storage()[i*stride()] for i=1,...,size()-1
	difference_type stride()const{
		return expression().stride();
	}
	
	// ---------
	// High level interface
	// ---------

	// Element access
	reference operator()(index_type i) const{
		return m_expression(m_range(i));
	}

	// Assignment operators 
	vector_range& operator = (vector_range const& vr){
		return assign(*this, typename vector_temporary<V>::type(vr));
	}

	template<class E>
	vector_range& operator = (vector_expression<E> const& e){
		return assign(*this, typename vector_temporary<E>::type(e));
	}

	typedef subrange_iterator< typename vector_closure_type::iterator> iterator;
	typedef subrange_iterator< typename vector_closure_type::const_iterator> const_iterator;

	const_iterator begin() const{
		return const_iterator(
		        m_expression.begin(),m_expression.end(),
		        start(),start()
		);
	}
	iterator begin(){
		return iterator(
		        m_expression.begin(),m_expression.end(),
		        start(),start()
		);
	}
	const_iterator end() const{
		return const_iterator(
		        m_expression.begin(),m_expression.end(),
		        start()+size(),start()
		);
	}
	iterator end(){
		return iterator(
		        m_expression.begin(),m_expression.end(),
		        start()+size(),start()
		);
	}
	
	void clear(){
		clear_range(begin(),end());
	}
	
	iterator set_element(iterator pos, index_type index, value_type value){
		return iterator(m_expression.set_element(pos.inner(),index+start(),value),start());
	}
	
	iterator clear_range(iterator first, iterator last){
		return iterator(m_expression.clear_range(first.inner(),last.inner()),start());
	}

	iterator clear_element(iterator pos){
		return iterator(m_expression.clear_element(pos.inner()),start());
	}
	
	void reserve(size_type non_zeros){
		m_expression.reserve(non_zeros);
	}
private:
	vector_closure_type m_expression;
	blas::range m_range;
};

// ------------------
// Simple Projections
// ------------------

/** \brief Return a \c vector_range on a specified vector, a start and stop index.
 * Return a \c vector_range on a specified vector, a start and stop index. The resulting \c vector_range can be manipulated like a normal vector.
 * If the specified range falls outside that of of the index range of the vector, then the resulting \c vector_range is not a well formed
 * Vector Expression and access to an element outside of index range of the vector is \b undefined.
 */
template<class V>
temporary_proxy<vector_range<V> > subrange(vector_expression<V>& data, typename V::size_type start, typename V::size_type stop){
	return vector_range<V> (data(), range(start, stop));
}

/** \brief Return a \c const \c vector_range on a specified vector, a start and stop index.
 * Return a \c const \c vector_range on a specified vector, a start and stop index. The resulting \c const \c vector_range can be manipulated like a normal vector.
 *If the specified range falls outside that of of the index range of the vector, then the resulting \c vector_range is not a well formed
 * Vector Expression and access to an element outside of index range of the vector is \b undefined.
 */
template<class V>
vector_range<typename const_expression<V>::type > subrange(
	vector_expression<V> const& data,
	typename V::size_type start,
	typename V::size_type stop
){
	return vector_range<typename const_expression<V>::type> (data(), range(start, stop));
}

template<class V>
temporary_proxy<vector_range<V> > subrange(temporary_proxy<V> data, typename V::size_type start, typename V::size_type stop){
	return subrange(static_cast<V&>(data), start, stop);
}

/// \brief Represents a given chunk of memory as a dense vector of elements of type T.
///
/// This adaptor is read/write if T is non-const and read-only if T is const.
template<class T>
class dense_vector_adaptor: public vector_expression<dense_vector_adaptor<T> > {
	typedef dense_vector_adaptor<T> self_type;
public:

	//std::container types
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

	typedef vector_reference<self_type const> const const_closure_type;
	typedef vector_reference<self_type> closure_type;
	typedef dense_tag storage_category;
	typedef elementwise_tag evaluation_category;

	// Construction and destruction

	/// \brief Constructor of a self_type proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression The Expression from which to construct the Proxy
 	template<class E>
	dense_vector_adaptor(vector_expression<E> const& expression)
	: m_values(expression().storage())
	, m_size(expression().size())
	, m_stride(expression().stride()){}
	
	/// \brief Constructor of a self_type proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression The Expression from which to construct the Proxy
 	template<class E>
	dense_vector_adaptor(vector_expression<E>& expression)
	: m_values(expression().storage())
	, m_size(expression().size())
	, m_stride(expression().stride()){}
		
	/// \brief Constructor of a self_type proxy from a block of memory
	/// \param values the block of memory used
	/// \param size size of the self_type
 	/// \param stride distance between elements of the self_type in memory
	dense_vector_adaptor(pointer values, size_type size, difference_type stride = 1 ):
		m_values(values),m_size(size),m_stride(stride){}	

	/// \brief Copy-constructor of a self_type
	/// \param v is the proxy to be copied
	template<class U>
	dense_vector_adaptor(dense_vector_adaptor<U> const& v)
	:m_values(v.storage()),m_size(v.size()),m_stride(v.stride())
	{}
		
	// ---------
	// Dense low level interface
	// ---------
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_size;
	}
	
	///\brief Returns the pointer to the beginning of the vector storage
	///
	/// Low-level access to the vectors internals. Elements storage()[i*stride()] for i=1,...,size()-1 are valid
	pointer storage()const{
		return m_values;
	}
	
	///\brief Returns th stride between the elements in storage()
	///
	/// In general elements of dense storage entities are spaced like storage()[i*stride()] for i=1,...,size()-1
	difference_type stride()const{
		return m_stride;
	}
	
	// ---------
	// High level interface
	// ---------
	
	// --------------
	// Element access
	// --------------

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator()(index_type i) const {
		return m_values[i*m_stride];
	}
	
	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator()(index_type i) {
		return m_values[i*m_stride];
	}	

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator[](index_type i) const {
		return m_values[i*m_stride];
	}
	
	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator[](index_type i) {
		return m_values[i*m_stride];
	}

	// ------------------
	// Element assignment
	// ------------------
	
	/// \brief Set element \f$i\f$ to the value \c t
	/// \param i index of the element
	/// \param t reference to the value to be set
	reference insert_element(index_type i, const_reference t) {
		return(*this)[i] = t;
	}

	/// \brief Set element \f$i\f$ to the \e zero value
	/// \param i index of the element
	void erase_element(index_type i) {
		(*this)[i] = value_type/*zero*/();
	}
		

	dense_vector_adaptor& operator = (dense_vector_adaptor const& e) {
		return assign(typename vector_temporary<self_type>::type(e));
	}
	template<class E>
	dense_vector_adaptor& operator = (const vector_expression<E>& e) {
		return assign(typename vector_temporary<E>::type(e));
	}
	
	// --------------
	// ITERATORS
	// --------------
	

	typedef dense_storage_iterator<T> iterator;
	typedef dense_storage_iterator<value_type const> const_iterator;

	/// \brief return an iterator on the first element of the vector
	const_iterator begin() const {
		return const_iterator(m_values,0);
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator end() const {
		return const_iterator(m_values+size()*stride(),size());
	}

	/// \brief Return an iterator on the first element of the vector
	iterator begin(){
		return iterator(m_values,0);
	}

	/// \brief Return an iterator at the end of the vector
	iterator end(){
		return iterator(m_values+size()*stride(),size());
	}
	
	//insertion and erasing of elements
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
		RANGE_CHECK(start < end);
		for(; start != end; ++start){
			*start = value_type/*zero*/();
		}
		return end;
	}
private:
	pointer m_values;
	std::size_t m_size;
	std::ptrdiff_t m_stride;
};

/// \brief Converts a chunk of memory into a vector of a given size.
template <class T>
temporary_proxy<dense_vector_adaptor<T> > adapt_vector(std::size_t size, T * data){
	return dense_vector_adaptor<T>(data,size);
}

/// \brief Converts a C-style array into a vector.
template <class T, std::size_t N>
temporary_proxy<dense_vector_adaptor<T> > adapt_vector(T (&array)[N]){
	return dense_vector_adaptor<T>(array,N);
}

template<class T,class I>
class sparse_vector_adaptor: public vector_expression<sparse_vector_adaptor<T,I> > {
	typedef sparse_vector_adaptor<T,I> self_type;
public:

	//std::container types
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename boost::remove_const<T>::type value_type;
	typedef value_type scalar_type;
	typedef value_type const& const_reference;
	typedef const_reference  reference;
	typedef value_type const* const_pointer;
	typedef const_pointer pointer;
	
	typedef typename boost::remove_const<I>::type index_type;
	typedef index_type const* const_index_pointer;
	typedef const_index_pointer index_pointer;

	typedef sparse_tag storage_category;
	typedef elementwise_tag evaluation_category;
	typedef vector_reference<self_type const> const const_closure_type;
	typedef vector_reference<self_type> closure_type;

	// Construction and destruction

	/// \brief Constructor of a self_type proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
 	template<class E>
	sparse_vector_adaptor(vector_expression<E> const& expression)
	: m_nonZeros(expression().nnz())
	, m_indices(expression().indices())
	, m_values(expression().values())
	, m_size(expression().size()){}
	
	
	sparse_vector_adaptor():m_size(0){}
		
	/// \brief Constructor of a vector proxy from a block of memory
	/// \param size the size of the vector represented by the memory
	/// \param values the block of memory used to store the values
	/// \param indices the block of memory used to store the indices
	/// \param memoryLength length of the strip of memory
	sparse_vector_adaptor(
		size_type size, const_pointer values,
		const_index_pointer indices, 
		size_type memoryLength
	): m_nonZeros(memoryLength)
	, m_indices(indices)
	, m_values(values)
	, m_size(size){}
	
	/// \brief Return the size of the vector
	size_type size() const {
		return m_size;
	}
	
	// ---------
	// Sparse low level interface
	// ---------
	
	/// \brief Number of nonzero elements of the vector.
	size_type nnz()const{
		return m_nonZeros;
	}
	/// \brief Array of values of the nonzero elements.
	const_pointer values()const{
		return m_values;
	}
	/// \brief Array of indices of the nonzero elements.
	index_pointer indices()const{
		return m_indices;
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	value_type operator()(index_type i) const {
		SIZE_CHECK(i < m_size);
		const_index_pointer pos = std::lower_bound(indices(),indices()+nnz(), i);
		difference_type diff = pos-indices();
		if(diff == (difference_type) nnz() || *pos != i)
			return value_type();
		return values()[diff];
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	value_type operator[](index_type i) const {
		return (*this)(i);
	}

	// --------------
	// ITERATORS
	// --------------
	
	typedef compressed_storage_iterator<value_type const, index_type const> const_iterator;
	typedef const_iterator iterator;

	/// \brief return an iterator behind the last non-zero element of the vector
	const_iterator begin() const {
		return const_iterator(values(),indices(),0);
	}

	/// \brief return an iterator behind the last non-zero element of the vector
	const_iterator end() const {
		return const_iterator(values(),indices(),nnz());
	}
private:
	std::size_t m_nonZeros;
	const_index_pointer m_indices;
	const_pointer m_values;

	std::size_t m_size;
};


}
}

#endif
