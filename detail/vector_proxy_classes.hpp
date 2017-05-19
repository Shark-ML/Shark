/*!
 * \brief       Classes used for vector proxies
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
 #ifndef REMORA_VECTOR_PROXY_CLASSES_HPP
#define REMORA_VECTOR_PROXY_CLASSES_HPP

#include "../cpu/iterator.hpp"
#include "traits.hpp"

#include <type_traits>
namespace remora{

template<class V>
class vector_reference:public vector_expression<vector_reference<V>, typename V::device_type >{
public:
	typedef typename V::size_type size_type;
	typedef typename V::value_type value_type;
	typedef typename V::const_reference const_reference;
	typedef typename reference<V>::type reference;

	typedef vector_reference<V const> const_closure_type;
	typedef vector_reference<V> closure_type;
	typedef typename storage<V>::type storage_type;
	typedef typename V::const_storage_type const_storage_type;
	typedef typename V::evaluation_category evaluation_category;

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

	storage_type raw_storage()const{
		return expression().raw_storage();
	}
	
	typename device_traits<typename V::device_type>::queue_type& queue()const{
		return m_expression->queue();
	}
	
	// ---------
	// High level interface
	// ---------
	
	// Element access
	template <class IndexExpression>
	auto operator()(IndexExpression const& i) const -> decltype(std::declval<V&>()(i)){
		return expression()(i);
	}
	template <class IndexExpression>
	auto operator[](IndexExpression const& i) const -> decltype(std::declval<V&>()(i)){
		return expression()(i);
	}
	
	vector_reference& operator = (vector_reference const& v){
		expression() = v.expression();
		return *this;
	}
	template<class E>
	vector_reference& operator = (vector_expression<E, typename V::device_type> const& e){
		expression() = e();
		return *this;
	}

	// Iterator types
	typedef typename V::const_iterator const_iterator;
	typedef typename  std::conditional<
		std::is_const<V>::value,
		typename V::const_iterator,
		typename V::iterator
	>::type iterator;

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
	iterator set_element(iterator pos, size_type index, value_type value){
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
 * \tparam V the type of vector referenced (for example \c vector<double>)
 */
template<class V>
class vector_range:public vector_expression<vector_range<V>, typename V::device_type >{
public:
	typedef typename V::size_type size_type;
	typedef typename V::value_type value_type;
	typedef typename V::const_reference const_reference;
	typedef typename reference<V>::type reference;

	typedef typename closure<V>::type vector_closure_type;
	typedef vector_range<typename const_expression<V>::type> const_closure_type;
	typedef vector_range<V> closure_type;
	typedef typename storage<V>::type storage_type;
	typedef typename V::const_storage_type const_storage_type;
	typedef typename V::evaluation_category evaluation_category;

	// Construction and destruction
	vector_range(vector_closure_type const& data, size_type start, size_type end):
		m_expression(data), m_start(start), m_size(end-start){
		REMORA_RANGE_CHECK(start <= end);
		REMORA_RANGE_CHECK(start <= m_expression.size());
		REMORA_RANGE_CHECK(end <= m_expression.size());
	}
	
	//non-const-> const conversion
	template<class E>
	vector_range(
		vector_range<E> const& other,
		typename std::enable_if<
			!std::is_same<E,vector_range>::value
		>::type* dummy = 0
	):m_expression(other.expression())
	, m_start(other.start()), m_size(other.size()){}
	
	// ---------
	// Internal Accessors
	// ---------
	
	size_type start() const{
		return m_start;
	}
	
	vector_closure_type const& expression() const{
		return m_expression;
	}
	vector_closure_type& expression(){
		return m_expression;
	}
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_size;
	}

	storage_type raw_storage()const{
		return expression().raw_storage().sub_region(start());
	}

	typename device_traits<typename V::device_type>::queue_type& queue()const{
		return m_expression.queue();
	}
	
	// ---------
	// High level interface
	// ---------

	// Element access	
	template <class IndexExpr>
	auto operator()(IndexExpr const& i) const -> decltype(std::declval<V&>()(
		device_traits<typename V::device_type>::index_add(std::size_t(),i)
	)){
		return m_expression(
			device_traits<typename V::device_type>::index_add(start(),i)
		);
	}

	// Assignment operators 
	vector_range& operator = (vector_range const& vr){
		return assign(*this, typename vector_temporary<V>::type(vr));
	}

	template<class E>
	vector_range& operator = (vector_expression<E, typename V::device_type> const& e){
		return assign(*this, typename vector_temporary<E>::type(e));
	}

	typedef typename device_traits<typename V::device_type>:: 
		template subrange_iterator< typename vector_closure_type::iterator>::type iterator;
	typedef typename device_traits<typename V::device_type>:: 
		template subrange_iterator< typename vector_closure_type::const_iterator>::type const_iterator;

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
	
	iterator set_element(iterator pos, size_type index, value_type value){
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
	size_type m_start;
	size_type m_size;
};

/// \brief Represents a given chunk of memory as a dense vector of elements of type T.
///
/// This adaptor is read/write if T is non-const and read-only if T is const.
template<class T, class Tag = cpu_tag>
class dense_vector_adaptor: public vector_expression<dense_vector_adaptor<T, Tag>, Tag > {
	typedef dense_vector_adaptor<T, Tag> self_type;
public:

	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type const& const_reference;
	typedef T&  reference;

	typedef dense_vector_adaptor<T const> const_closure_type;
	typedef dense_vector_adaptor closure_type;
	typedef dense_vector_storage<T> storage_type;
	typedef dense_vector_storage<value_type const> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction

	/// \brief Constructor of a self_type proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression The Expression from which to construct the Proxy
 	template<class E>
	dense_vector_adaptor(vector_expression<E, cpu_tag> const& expression)
	: m_values(expression().raw_storage().values)
	, m_size(expression().size())
	, m_stride(expression().raw_storage().stride){}
	
	/// \brief Constructor of a self_type proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression The Expression from which to construct the Proxy
 	template<class E>
	dense_vector_adaptor(vector_expression<E,cpu_tag>& expression)
	: m_values(expression().raw_storage().values)
	, m_size(expression().size())
	, m_stride(expression().raw_storage().stride){}
		
	/// \brief Constructor of a self_type proxy from a block of memory
	/// \param values the block of memory used
	/// \param size size of the self_type
 	/// \param stride distance between elements of the self_type in memory
	dense_vector_adaptor(T* values, size_type size, size_type stride = 1 ):
		m_values(values),m_size(size),m_stride(stride){}	

	/// \brief Copy-constructor of a self_type
	/// \param v is the proxy to be copied
	template<class U>
	dense_vector_adaptor(dense_vector_adaptor<U> const& v)
	: m_values(v.raw_storage().values)
	, m_size(v.size())
	, m_stride(v.raw_storage().stride){}
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_size;
	}
	
	///\brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage() const{
		return {m_values,m_stride};
	}
	
	typename device_traits<cpu_tag>::queue_type& queue(){
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

	// ------------------
	// Element assignment
	// ------------------
	
	/// \brief Set element \f$i\f$ to the value \c t
	/// \param i index of the element
	/// \param t reference to the value to be set
	reference insert_element(size_type i, const_reference t) {
		return(*this)[i] = t;
	}

	/// \brief Set element \f$i\f$ to the \e zero value
	/// \param i index of the element
	void erase_element(size_type i) {
		(*this)[i] = value_type/*zero*/();
	}
		

	dense_vector_adaptor& operator = (dense_vector_adaptor const& e) {
		return assign(typename vector_temporary<self_type>::type(e));
	}
	template<class E>
	dense_vector_adaptor& operator = (vector_expression<E, cpu_tag> const& e) {
		return assign(typename vector_temporary<E>::type(e));
	}
	
	// --------------
	// ITERATORS
	// --------------
	

	typedef iterators::dense_storage_iterator<T> iterator;
	typedef iterators::dense_storage_iterator<value_type const> const_iterator;

	/// \brief return an iterator on the first element of the vector
	const_iterator begin() const {
		return const_iterator(m_values,0);
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator end() const {
		return const_iterator(m_values+size()*m_stride,size());
	}

	/// \brief Return an iterator on the first element of the vector
	iterator begin(){
		return iterator(m_values,0);
	}

	/// \brief Return an iterator at the end of the vector
	iterator end(){
		return iterator(m_values+size()*m_stride,size());
	}
	
	//insertion and erasing of elements
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
		REMORA_RANGE_CHECK(start < end);
		for(; start != end; ++start){
			*start = value_type/*zero*/();
		}
		return end;
	}
private:
	T* m_values;
	size_type m_size;
	size_type m_stride;
};

template<class T,class I>
class sparse_vector_adaptor: public vector_expression<sparse_vector_adaptor<T,I>, cpu_tag > {
	typedef sparse_vector_adaptor<T,I> self_type;
public:

	//std::container types
	typedef typename std::remove_const<I>::type size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;
	
	typedef sparse_vector_adaptor<value_type const,size_type const> const_closure_type;
	typedef sparse_vector_adaptor closure_type;
	typedef sparse_vector_storage<T const,I const> storage_type;
	typedef sparse_vector_storage<value_type const,size_type const> const_storage_type;
	typedef elementwise<sparse_tag> evaluation_category;

	// Construction and destruction

	/// \brief Constructor of a self_type proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
 	template<class E>
	sparse_vector_adaptor(vector_expression<E, cpu_tag> const& expression)
	: m_nonZeros(expression().raw_storage().nnz)
	, m_indices(expression().raw_storage().indices)
	, m_values(expression().raw_storage().values)
	, m_size(expression().size()){}
	
	
	sparse_vector_adaptor():m_size(0){}
		
	/// \brief Constructor of a vector proxy from a block of memory
	/// \param size the size of the vector represented by the memory
	/// \param values the block of memory used to store the values
	/// \param indices the block of memory used to store the indices
	/// \param memoryLength length of the strip of memory
	sparse_vector_adaptor(
		size_type size, 
		value_type const* values,
		size_type const* indices, 
		size_type memoryLength
	): m_nonZeros(memoryLength)
	, m_indices(indices)
	, m_values(values)
	, m_size(size){}
	
	/// \brief Return the size of the vector
	size_type size() const {
		return m_size;
	}
	
	///\brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage() const{
		return {m_values,m_indices, m_nonZeros};
	}
	
	typename device_traits<cpu_tag>::queue_type& queue(){
		return device_traits<cpu_tag>::default_queue();
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	value_type operator()(size_type i) const {
		REMORA_SIZE_CHECK(i < m_size);
		size_type const* pos = std::lower_bound(m_indices,m_indices+m_nonZeros, i);
		std::ptrdiff_t diff = pos-m_indices;
		if(diff == (std::ptrdiff_t) m_nonZeros || *pos != i)
			return value_type();
		return m_values[diff];
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	value_type operator[](size_type i) const {
		return (*this)(i);
	}

	// --------------
	// ITERATORS
	// --------------
	
	typedef iterators::compressed_storage_iterator<value_type const, size_type const> const_iterator;
	typedef const_iterator iterator;

	/// \brief return an iterator behind the last non-zero element of the vector
	const_iterator begin() const {
		return const_iterator(m_values,m_indices,0);
	}

	/// \brief return an iterator behind the last non-zero element of the vector
	const_iterator end() const {
		return const_iterator(m_values,m_indices,m_nonZeros);
	}
private:
	std::size_t m_nonZeros;
	size_type const* m_indices;
	value_type const* m_values;

	std::size_t m_size;
};

}
#endif
