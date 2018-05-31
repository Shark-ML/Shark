/*!
 * \brief       Iterators for elementwise vector expression evaluation
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
#ifndef REMORA_DETAIL_ITERATOR_HPP
#define REMORA_DETAIL_ITERATOR_HPP

#include <iterator>
#include <type_traits>
#include <limits>
#include <algorithm>
#include <cstdlib>

#include "../detail/check.hpp"

namespace remora{ namespace iterators{

// Iterator tags -- hierarchical definition of storage characteristics
struct sparse_random_access_iterator_tag: public std::random_access_iterator_tag{};
struct packed_random_access_iterator_tag: public std::random_access_iterator_tag{};
struct dense_random_access_iterator_tag: public packed_random_access_iterator_tag{};

/** \brief Base class of all random access iterators.
 *
 * \param I the derived iterator type
 * \param T the value type
 * \param Tag the iterator tag - must be dense/packed_random_access_iterator_tag
 *
 * The random access iterator can proceed in both directions
 * via the post increment/decrement operator or in larger steps
 * via the +, - and +=, -= operators. The random access iterator
 * is LessThanComparable.
 */
template<class I, class T, class Tag>
struct random_access_iterator_base
:public std::iterator<Tag, T> {
	typedef I derived_iterator_type;
	typedef T derived_value_type;
	typedef std::ptrdiff_t difference_type;

	friend I operator + (random_access_iterator_base const& it, difference_type n) {
		I tmp(static_cast<const I&>(it));
		return tmp += n;
	}
	friend I operator - (random_access_iterator_base const& it, difference_type n) {
		I tmp(static_cast<const I&>(it));
		return tmp -= n;
	}
	
	friend I operator + (difference_type n, random_access_iterator_base const& it) {
		I tmp(static_cast<const I&>(it));
		return tmp += n;
	}
	friend I operator - (difference_type n, random_access_iterator_base const& it) {
		I tmp(static_cast<const I&>(it));
		return tmp -= n;
	}
};
//arithmetic for random_access_iterators
template<class I, class T, class Tag>
I operator ++ (random_access_iterator_base<I,T,Tag>& it,int) {
	I& d = static_cast<I&>(it);
	I tmp(d);
	++ d;
	return tmp;
}

template<class I, class T, class Tag>
I operator -- (random_access_iterator_base<I,T,Tag>& it,int) {
	I& d = static_cast<I&>(it);
	I tmp(d);
	-- d;
	return tmp;
}


// Comparison of random_access_iterators
template<class I1, class T1, class I2, class T2, class Tag>
bool operator != (
	random_access_iterator_base<I1,T1,Tag> const& it1, 
	random_access_iterator_base<I2,T2,Tag> const& it2
){
	I1 const& d1 = static_cast<const I1&>(it1);
	I2 const& d2 = static_cast<const I2&>(it2);
	
	return !(d1 == d2);
}
template<class I1, class T1, class I2, class T2, class Tag>
bool operator <= (
	random_access_iterator_base<I1,T1,Tag> const& it1, 
	random_access_iterator_base<I2,T2,Tag> const& it2
){
	I1 const& d1 = static_cast<const I1&>(it1);
	I2 const& d2 = static_cast<const I2&>(it2);
	
	return !(d2 < d1);
}
template<class I1, class T1, class I2, class T2, class Tag>
bool operator >= (
	random_access_iterator_base<I1,T1,Tag> const& it1, 
	random_access_iterator_base<I2,T2,Tag> const& it2
){
	I1 const& d1 = static_cast<const I1&>(it1);
	I2 const& d2 = static_cast<const I2&>(it2);
	
	return !(d1 < d2);
}
template<class I1, class T1, class I2, class T2, class Tag>
bool operator > (
	random_access_iterator_base<I1,T1,Tag> const& it1, 
	random_access_iterator_base<I2,T2,Tag> const& it2
){
	I1 const& d1 = static_cast<const I1&>(it1);
	I2 const& d2 = static_cast<const I2&>(it2);

	return d2 < d1;
}
//traits lass for choosing the right base for wrapping iterators

template<class Closure>
class indexed_iterator:
	public random_access_iterator_base<
		indexed_iterator<Closure>,
		typename Closure::value_type,
		dense_random_access_iterator_tag
	> {
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename Closure::value_type value_type;
	typedef typename std::conditional<
		std::is_const<Closure>::value,
		typename Closure::const_reference,
		typename Closure::reference
	>::type reference;

	// Construction and destruction
	indexed_iterator(){}
	indexed_iterator(Closure container, size_type index)
	: m_index(index), m_closure(container) {}
		
	template<class C>
	indexed_iterator(indexed_iterator<C> const& iterator)
	: m_index(iterator.m_index), m_closure(iterator.m_closure) {}

	// Arithmetic
	indexed_iterator& operator++() {
		++m_index;
		return *this;
	}
	indexed_iterator& operator--() {
		--m_index;
		return *this;
	}
	indexed_iterator& operator += (difference_type n) {
		m_index += n;
		return *this;
	}
	indexed_iterator& operator -= (difference_type n) {
		m_index -= n;
		return *this;
	}
	template<class T>
	difference_type operator - (indexed_iterator<T> const& it) const {
		return m_index - it.m_index;
	}

	// Dereference
	reference operator *() const {
		REMORA_RANGE_CHECK(m_index < m_closure.size());
		return m_closure(m_index);
	}
	reference operator [](difference_type n) const {
		REMORA_RANGE_CHECK(m_index+n < m_closure.size());
		return m_closure(m_index+n);
	}

	// Index
	size_type index() const {
		return m_index;
	}

	// Assignment
	template<class T>
	indexed_iterator &operator = (indexed_iterator<T> const& it) {
		m_closure = it.m_closure;
		m_index = it.m_index;
		return *this;
	}

	// Comparison
	template<class T>
	bool operator == (indexed_iterator<T> const& it) const {
		return m_index == it.m_index;
	}
	template<class T>
	bool operator < (indexed_iterator<T> const& it) const {
		return m_index < it.m_index;
	}

private:
	size_type m_index;
	Closure m_closure;
	template<class> friend class indexed_iterator;
};

template<class T, class Tag=dense_random_access_iterator_tag>
class dense_storage_iterator:
public random_access_iterator_base<
	dense_storage_iterator<T,Tag>,
	typename std::remove_const<T>::type, 
	Tag
>{
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef T& reference;
	typedef T* pointer;

	// Construction
	dense_storage_iterator():m_pos(0),m_index(0) {}
	dense_storage_iterator(pointer pos, size_type index, difference_type stride = 1)
	:m_pos(pos), m_index(index), m_stride(stride) {}
	
	template<class U>
	dense_storage_iterator(dense_storage_iterator<U,Tag> const& iter)
	:m_pos(iter.m_pos), m_index(iter.m_index), m_stride(iter.m_stride){}
		
	template<class U>
	dense_storage_iterator& operator=(dense_storage_iterator<U,Tag> const& iter){
		m_pos = iter.m_pos;
		m_index = iter.m_index;
		m_stride = iter.m_stride;
		return *this;
	}

	// Arithmetic
	dense_storage_iterator& operator ++ () {
		++m_index;
		m_pos += m_stride;
		return *this;
	}
	dense_storage_iterator& operator -- () {
		--m_index;
		m_pos -= m_stride;
		return *this;
	}
	dense_storage_iterator& operator += (difference_type n) {
		m_index += n;
		m_pos += n*m_stride;
		return *this;
	}
	dense_storage_iterator& operator -= (difference_type n) {
		m_index -= n;
		m_pos -= n*m_stride;
		return *this;
	}
	template<class U>
	difference_type operator - (dense_storage_iterator<U,Tag> const& it) const {
		//REMORA_RANGE_CHECK(m_pos == it.m_pos);
		return (difference_type)m_index - (difference_type)it.m_index;
	}

	// Dereference
	reference operator*() const {
		return *m_pos;
	}
	reference operator [](difference_type n) const {
		return m_pos[m_stride*n];
	}

	// Index
	size_type index() const {
		return m_index;
	}

	// Comparison
	template<class U>
	bool operator == (dense_storage_iterator<U,Tag> const& it) const {
		//REMORA_RANGE_CHECK(m_pos == it.m_pos);
		//~ REMORA_RANGE_CHECK(m_index < it.m_index);
		return m_index == it.m_index;
	}
	template<class U>
	bool operator <  (dense_storage_iterator<U,Tag> const& it) const {
		//REMORA_RANGE_CHECK(m_pos == it.m_pos);
		return m_index < it.m_index;
	}

private:
	pointer m_pos;
	size_type m_index;
	difference_type m_stride;
	template<class,class> friend class dense_storage_iterator;
};
template<class T, class I>
class compressed_storage_iterator:
public random_access_iterator_base<
	compressed_storage_iterator<T,I>,
	typename std::remove_const<T>::type, 
	sparse_random_access_iterator_tag
>{
public:
	typedef typename std::remove_const<T>::type value_type;
	typedef typename std::remove_const<I>::type size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T& reference;
	typedef T* pointer;

	// Construction and Assignment
	compressed_storage_iterator() {}
	compressed_storage_iterator(
		T* value_array, I* index_array, 
		size_type position, size_type major_pos = 0
	)
	: m_values(value_array),m_indices(index_array)
	, m_position(position), m_major_pos(major_pos){}
		
	template<class U,class V>
	compressed_storage_iterator(compressed_storage_iterator<U,V> const& it) {
		m_position = it.m_position;
		m_major_pos = it.m_major_pos;
		m_values = it.m_values;
		m_indices = it.m_indices;
	}

	template<class U,class V>
	compressed_storage_iterator &operator = (compressed_storage_iterator<U,V> const& it) {
		m_position = it.m_position;
		m_major_pos = it.m_major_pos;
		m_values = it.m_values;
		m_indices = it.m_indices;
		return *this;
	}

	// Arithmetic
	compressed_storage_iterator &operator++ () {
		++m_position;
		return *this;
	}
	compressed_storage_iterator &operator -- () {
		REMORA_RANGE_CHECK(m_position > 0);
		--m_position;
		return *this;
	}
	
	compressed_storage_iterator &operator += (difference_type n) {
		m_position += n;
		return *this;
	}
	compressed_storage_iterator &operator -= (difference_type n) {
		m_position -= n;
		return *this;
	}

	// Dereference
	reference operator* () const {
		return m_values[m_position];
	}
	reference operator [](difference_type n) const {
		return m_values[m_position+n];
	}
	size_type index() const {
		return m_indices[m_position];
	}
	
	template<class U,class V>
	difference_type operator - (compressed_storage_iterator<U,V> const& it) const {
		REMORA_RANGE_CHECK(m_values == it.m_values);
		REMORA_RANGE_CHECK(m_indices == it.m_indices);
		return difference_type(m_position) - difference_type(it.m_position);
	}
	
	size_type major_index()const{
		return m_major_pos;
	}

	template<class U,class V>
	bool operator == (compressed_storage_iterator<U,V> const &it) const {
		REMORA_RANGE_CHECK(m_values == it.m_values);
		REMORA_RANGE_CHECK(m_indices == it.m_indices);
		return m_position == it.m_position;
	}

private:
	T* m_values;
	I* m_indices;
	std::size_t m_position;
	std::size_t m_major_pos;
	template<class,class> friend class compressed_storage_iterator;
};



template<class T>
class constant_iterator:
public random_access_iterator_base<
	constant_iterator<T>,
	typename std::remove_const<T>::type,
	dense_random_access_iterator_tag
>{
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef value_type const &reference;
	typedef value_type const *pointer;

	// Construction and destruction
	constant_iterator() {}
	constant_iterator(value_type value, size_type position)
		:m_position(position),m_value(value) {}

	// Arithmetic
	constant_iterator &operator ++ () {
		++ m_position;
		return *this;
	}
	constant_iterator &operator -- () {
		-- m_position;
		return *this;
	}
	constant_iterator &operator += (difference_type n) {
		m_position += n;
		return *this;
	}
	constant_iterator &operator -= (difference_type n) {
		m_position -= n;
		return *this;
	}
	difference_type operator - (constant_iterator const &it) const {
		return m_position - it.m_position;
	}

	// Dereference
	reference operator * () const {
		return m_value;
	}
	reference operator [](difference_type n) const {
		return m_value;
	}

	// Indices
	size_type index() const {
		return m_position;
	}

	// Assignment
	template<class Iter>
	constant_iterator &operator = (constant_iterator const &it) {
		m_position = it.m_position;
		m_value = it.m_value;
		return *this;
	}

	// Comparison
	bool operator == (constant_iterator const &it) const {
		return m_position == it.m_position;
	}
	bool operator < (constant_iterator const &it) const {
		return m_position < it.m_position;
	}
private:
	size_type m_position;
	value_type m_value;
};

template<class BaseIterator, class F>
class transform_iterator:
public random_access_iterator_base<
	transform_iterator<BaseIterator,F>, 
	typename BaseIterator::value_type,
	typename BaseIterator::iterator_category
>{
public:
	typedef typename BaseIterator::iterator_category iterator_category;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename std::result_of<F(typename BaseIterator::value_type)>::type value_type;
	typedef value_type reference;
	typedef value_type *pointer;

	// Construction and destruction
	transform_iterator() {}
	transform_iterator(BaseIterator const &it,F functor)
		:m_position(it),m_functor(functor) {}

	// Arithmetic
	transform_iterator &operator ++ () {
		++ m_position;
		return *this;
	}
	transform_iterator &operator -- () {
		-- m_position;
		return *this;
	}
	transform_iterator &operator += (difference_type n) {
		m_position += n;
		return *this;
	}
	transform_iterator &operator -= (difference_type n) {
		m_position -= n;
		return *this;
	}
	difference_type operator - (transform_iterator const &it) const {
		return m_position - it.m_position;
	}

	// Dereference
	reference operator * () const {
		return m_functor(*m_position);
	}
	reference operator [](difference_type n) const {
		return *(*this + n);
	}

	// Indices
	size_type index() const {
		return m_position.index();
	}

	// Assignment
	template<class Iter>
	transform_iterator &operator = (transform_iterator<Iter,F> const &it) {
		m_position = it.m_position;
		m_functor = it.m_functor;
		return *this;
	}

	// Comparison
	bool operator == (transform_iterator const &it) const {
		return m_position == it.m_position;
	}
	bool operator < (transform_iterator const &it) const {
		return m_position < it.m_position;
	}

private:
	BaseIterator m_position;
	F m_functor;
};

template<class T>
class one_hot_iterator:public random_access_iterator_base<
	one_hot_iterator<T>, T, sparse_random_access_iterator_tag
> {
public:
	typedef T value_type;
	typedef std::ptrdiff_t difference_type;
	typedef std::size_t size_type;
	typedef T& reference;
	typedef T const& const_reference;
	typedef value_type const* pointer;

	// Construction and destruction
	one_hot_iterator(){}
	one_hot_iterator(size_type index, value_type value, bool isEnd)
		:m_index(index),m_value(value),m_isEnd(isEnd){}

	// Arithmetic
	one_hot_iterator& operator ++ () {
		m_isEnd = true;
		return *this;
	}
	one_hot_iterator& operator -- () {
		m_isEnd = false;
		return *this;
	}
	one_hot_iterator &operator += (difference_type n) {
		m_isEnd += n;
		return *this;
	}
	one_hot_iterator &operator -= (difference_type n) {
		m_isEnd -= n;
		return *this;
	}
	

	// Dereference
	reference operator*() const {
		return m_value;
	}

	// Indices
	size_type index() const{
		return m_index;
	}
	difference_type operator - (one_hot_iterator const &it) const {
		return m_isEnd - it.m_isEnd;
	}

	// Assignment
	one_hot_iterator& operator = (one_hot_iterator const& it) {
		m_index = it.m_index;
		m_value = it.m_value;
		return *this;
	}

	// Comparison
	bool operator == (one_hot_iterator const& it) const {
		REMORA_RANGE_CHECK(m_index == it.m_index);
		return m_isEnd == it.m_isEnd;
	}
	bool operator < (one_hot_iterator const &it) const {
		return m_isEnd < it.m_isEnd;
	}

private:
	size_type m_index;
	value_type m_value;
	bool m_isEnd;
};


template<class I1, class I2>
struct iterator_restrict_traits {
	typedef I1 iterator_category;
};

template<>
struct iterator_restrict_traits<dense_random_access_iterator_tag, sparse_random_access_iterator_tag> {
	typedef sparse_random_access_iterator_tag iterator_category;
};

template<>
struct iterator_restrict_traits<packed_random_access_iterator_tag,dense_random_access_iterator_tag> {
	typedef dense_random_access_iterator_tag iterator_category;
};

template<>
struct iterator_restrict_traits<packed_random_access_iterator_tag,sparse_random_access_iterator_tag> {
	typedef sparse_random_access_iterator_tag iterator_category;
};

template<class Iterator1, class Iterator2, class F>
class binary_transform_iterator:
public random_access_iterator_base<
	binary_transform_iterator<Iterator1,Iterator2,F>,
	typename std::result_of<F(typename Iterator1::value_type, typename Iterator2::value_type)>::type,
	typename iterator_restrict_traits<
		typename Iterator1::iterator_category,
		typename Iterator2::iterator_category
	>::iterator_category
>{
private:
	typedef typename Iterator1::iterator_category category1;
	typedef typename Iterator2::iterator_category category2;
public:
	typedef typename iterator_restrict_traits<
		category1,
		category2
	>::iterator_category iterator_category;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename std::result_of<F(typename Iterator1::value_type, typename Iterator2::value_type)>::type value_type;
	typedef value_type reference;
	typedef value_type *pointer;

	// Construction and destruction
	binary_transform_iterator() {}
	binary_transform_iterator(
	    F functor,
	    Iterator1 const &it1, Iterator1 const &end1,
	    Iterator2 const &it2, Iterator2 const &end2
	):m_index(0)
	, m_iterator1(it1), m_end1(end1)
	, m_iterator2(it2), m_end2(end2)
	, m_functor(functor) 
	{
		//ensure that both iterators start at a valid location
		ensureValidPosition(category1(),category2());
		
		//we can't get the correct index for end iterators
		if(m_iterator1 != end1 && m_iterator2 != end2)
			m_index = std::min(m_iterator1.index(),m_iterator2.index());
		
		
	}

private:
	//we need to handle all specializations independently from each other
	
	// Dense specializations are easy
	void ensureValidPosition(
		dense_random_access_iterator_tag,
		dense_random_access_iterator_tag
	){}
	void increment(
		dense_random_access_iterator_tag,
		dense_random_access_iterator_tag
	) {
		++ m_index;
		++ m_iterator1;
		++ m_iterator2;
	}
	void decrement(
		dense_random_access_iterator_tag,
		dense_random_access_iterator_tag
	) {
		-- m_index;
		-- m_iterator1;
		-- m_iterator2;
	}
	void increment(
		dense_random_access_iterator_tag,
		dense_random_access_iterator_tag, 
		difference_type n
	) {
		m_index += n;
		m_iterator1 += n;
		m_iterator2 += n;
	}
	value_type dereference(
		dense_random_access_iterator_tag,
		dense_random_access_iterator_tag
	) const {
		return m_functor(*m_iterator1, *m_iterator2);
	}

	// Sparse specializations
	void ensureValidPosition(
		sparse_random_access_iterator_tag,
		sparse_random_access_iterator_tag
	){
		//ensure that both iterators point to the same index
		if(F::left_zero_remains || F::right_zero_remains){
			while(
				m_iterator1 != m_end1 && m_iterator2 != m_end2 
				&& m_iterator1.index() != m_iterator2.index()
			){
				if(m_iterator1.index() < m_iterator2.index())
					++m_iterator1;
				else
					++m_iterator2;
			}
			if( m_iterator1 == m_end1 || m_iterator2 == m_end2){
				m_iterator2 = m_end2;
				m_iterator1 = m_end1;
			}
		}
	}
	void increment(
		sparse_random_access_iterator_tag,
		sparse_random_access_iterator_tag
	) {
		if(F::left_zero_remains || F::right_zero_remains){
			++ m_iterator1;
			++ m_iterator2;
			ensureValidPosition(category1(),category2());
		}else{
			if (m_iterator1 != m_end1 && m_iterator2 != m_end2){
				if( m_iterator1.index() == m_iterator2.index()){
					++ m_iterator1;
					++ m_iterator2;
				}
				else if(m_iterator1.index() < m_iterator2.index())
					++m_iterator1;
				else
					++m_iterator2;
			}else if(m_iterator1 != m_end1){
				++ m_iterator1;
			}else{
				++ m_iterator2;
			}
		}
		size_type index1 = std::numeric_limits<size_type>::max();
		size_type index2 = std::numeric_limits<size_type>::max();
		if(m_iterator1 != m_end1)
			index1 = m_iterator1.index();
		if(m_iterator2 != m_end2)
			index2 = m_iterator2.index();
		
		m_index = std::min(index1, index2);
	}
	void decrement(
		sparse_random_access_iterator_tag,
		sparse_random_access_iterator_tag
	) {
		if (m_index <= m_iterator1.index())
			-- m_iterator1;
		if (m_index <= m_iterator2.index())
			-- m_iterator2;
		m_index = std::max(m_iterator1.index(), m_iterator2.index());
	}
	void increment(
		sparse_random_access_iterator_tag, 
		sparse_random_access_iterator_tag, 
		difference_type n
	) {
		while (n > 0) {
			increment(category1(),category2());
			--n;
		}
		while (n < 0) {
			decrement(category1(),category2());
			++n;
		}
	}
	value_type dereference(
		sparse_random_access_iterator_tag,
		sparse_random_access_iterator_tag
	) const {
		value_type t1 = value_type/*zero*/();
		if (m_iterator1 != m_end1 && m_iterator1.index() == m_index)
			t1 = *m_iterator1;
		value_type t2 = value_type/*zero*/();
		if (m_iterator2 != m_end2 && m_iterator2.index() == m_index)
			t2 = *m_iterator2;
		return m_functor(t1, t2);
	}
	
	//dense-sparse
	// Sparse specializations
	void ensureValidPosition(
		dense_random_access_iterator_tag,
		sparse_random_access_iterator_tag
	){
		if(F::right_zero_remains){
			m_iterator1 += m_iterator2.index()-m_iterator1.index();
		}
	}
	
	void increment(
		dense_random_access_iterator_tag,
		sparse_random_access_iterator_tag
	) {
		if(F::right_zero_remains){
			++ m_iterator2;
			m_iterator1 += m_iterator2.index()-m_iterator1.index();
		}else{
			if (m_iterator2 != m_end2){
				if( m_iterator1.index() == m_iterator2.index()){
					++ m_iterator1;
					++ m_iterator2;
				}
				else if(m_iterator2.index() > m_iterator1.index())
					++m_iterator1;
				else
					++m_iterator2;
			}else
				++ m_iterator1;
		}
		size_type index1 = m_iterator1.index();
		size_type index2 = std::numeric_limits<size_type>::max();
		if(m_iterator2 != m_end2)
			index2 = m_iterator2.index();
		m_index = std::min(index1, index2);
	}
	void decrement(
		dense_random_access_iterator_tag,
		sparse_random_access_iterator_tag
	) {
		if(F::right_zero_remains){
			-- m_iterator2;
			m_iterator1 -= m_iterator1.index()-m_iterator2.index();
		}else{
			if (m_index <= m_iterator1.index())
				-- m_iterator1;
			if (m_index <= m_iterator2.index())
				-- m_iterator2;
			m_index = std::max(m_iterator1.index(), m_iterator2.index());
		}
	}
	void increment(
		dense_random_access_iterator_tag, 
		sparse_random_access_iterator_tag, 
		difference_type n
	) {
		while (n > 0) {
			increment(category1(),category2());
			--n;
		}
		while (n < 0) {
			decrement(category1(),category2());
			++n;
		}
	}
	value_type dereference(
		dense_random_access_iterator_tag,
		sparse_random_access_iterator_tag
	) const {
		typedef typename Iterator2::value_type value_type2;
		value_type t2 = value_type2/*zero*/();
		if (m_iterator2 != m_end2 && m_iterator2.index() == m_index)
			t2 = *m_iterator2;
		return m_functor(*m_iterator1, t2);
	}
	
	//sparse-dense
	void ensureValidPosition(
		sparse_random_access_iterator_tag,
		dense_random_access_iterator_tag
	){
		if(F::left_zero_remains){
			m_iterator2 += m_iterator1.index()-m_iterator2.index();
		}
	}
	void increment(
		sparse_random_access_iterator_tag,
		dense_random_access_iterator_tag
	) {
		if(F::left_zero_remains){
			++ m_iterator1;
			m_iterator2 += m_iterator1.index()-m_iterator2.index();
		}else{
			if (m_iterator1 != m_end1){
				if( m_iterator1.index() == m_iterator2.index()){
					++ m_iterator1;
					++ m_iterator2;
				}
				else if(m_iterator1.index() > m_iterator2.index())
					++m_iterator2;
				else
					++m_iterator1;
			}else
				++ m_iterator2;
		}
		size_type index1 = std::numeric_limits<size_type>::max();
		size_type index2 = m_iterator2.index();
		if(m_iterator1 != m_end1)
			index1 = m_iterator1.index();
		m_index = std::min(index1, index2);
	}
	void decrement(
		sparse_random_access_iterator_tag,
		dense_random_access_iterator_tag
	) {
		if(F::left_zero_remains){
			-- m_iterator1;
			m_iterator2 -= m_iterator2.index()-m_iterator1.index();
		}else{
			if (m_index <= m_iterator2.index())
				-- m_iterator2;
			if (m_index <= m_iterator1.index())
				-- m_iterator1;
			m_index = std::max(m_iterator1.index(), m_iterator2.index());
		}
	}
	void increment(
		sparse_random_access_iterator_tag,
		dense_random_access_iterator_tag,
		difference_type n
	) {
		while (n > 0) {
			increment(category1(),category2());
			--n;
		}
		while (n < 0) {
			decrement(category1(),category2());
			++n;
		}
	}
	value_type dereference(
		sparse_random_access_iterator_tag,
		dense_random_access_iterator_tag
	) const {
		typedef typename Iterator1::value_type value_type1;
		value_type t1 = value_type1/*zero*/();
		if (m_iterator1 != m_end1 && m_iterator1.index() == m_index)
			t1 = *m_iterator1;
		return m_functor(t1,*m_iterator2);
	}

	public:
	// Arithmetic
	binary_transform_iterator &operator ++ () {
		increment(category1(),category2());
		return *this;
	}
	binary_transform_iterator &operator -- () {
		decrement(category1(),category2());
		return *this;
	}
	binary_transform_iterator &operator += (difference_type n) {
		increment(category1(),category2(), n);
		return *this;
	}
	binary_transform_iterator &operator -= (difference_type n) {
		increment(category1(),category2(), -n);
		return *this;
	}
	difference_type operator - (const binary_transform_iterator &it) const {
		difference_type diff1 = m_iterator1- it.m_iterator1;
		difference_type diff2 = m_iterator2- it.m_iterator2;
		return std::abs(diff1) > std::abs(diff2)? diff1:diff2;
	}

	// Dereference
	reference operator * () const {
		return dereference(category1(),category2());
	}
	reference operator [](difference_type n) const {
		return *(*this + n);
	}

	// Index
	size_type index() const {
		return m_index;
	}

	// Comparison
	bool operator == (binary_transform_iterator const &it) const {
		return m_iterator1 == it.m_iterator1 && m_iterator2 == it.m_iterator2;
	}
	bool operator < (binary_transform_iterator const &it) const {
		return m_iterator1 < it.m_iterator1 || m_iterator2 < it.m_iterator2;
	}
private:
	size_type m_index;
	Iterator1 m_iterator1;
	Iterator1 m_end1;
	Iterator2 m_iterator2;
	Iterator2 m_end2;
	F m_functor;
};

}}

#endif
