/*!
 * 
 *
 * \brief       Small Iterator collection.
 * 
 * 
 * 
 *
 * \author      Oswin Krause
 * \date        2012
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_CORE_ITERATORS_H
#define SHARK_CORE_ITERATORS_H

#include <boost/version.hpp>
#include "Impl/boost_iterator_facade_fixed.hpp"//thanks, boost.

#include <boost/range/iterator.hpp>
namespace shark{

///\brief creates an Indexed Iterator, an Iterator which also carries index information using index()
///
///The underlying Iterator must be a random access iterator
template<class Iterator>
class IndexedIterator: public  SHARK_ITERATOR_FACADE<
	IndexedIterator<Iterator>,
	typename boost::iterator_value<Iterator>::type,
	std::random_access_iterator_tag,
	typename boost::iterator_reference<Iterator>::type
>
{
public:
	IndexedIterator()
	:m_index(0){}

	///\brief Copy-Constructs this iterator from some other IndexedIterator convertible to this.
	template<class I>
	IndexedIterator(IndexedIterator<I> const& iterator)
	: m_iterator(iterator.m_iterator),m_index(iterator.index){}

	///\brief Constructs the iterator from another iterator plus a starting index.
	template<class IteratorT>
	IndexedIterator(IteratorT const& iterator, std::size_t startIndex)
	: m_iterator(Iterator(iterator)),m_index(startIndex){}

	std::size_t index()const{
		return m_index;
	}
private:
	typedef SHARK_ITERATOR_FACADE<
		IndexedIterator<Iterator>,
		typename boost::iterator_value<Iterator>::type,
		boost::random_access_traversal_tag,
		typename boost::iterator_reference<Iterator>::type
	> base_type;

	friend class SHARK_ITERATOR_CORE_ACCESS;

	typename base_type::reference dereference() const {
		return *m_iterator;
	}

	void increment() {
		++m_index;
		++m_iterator;
	}
	void decrement() {
		--m_index;
		--m_iterator;
	}

	void advance(std::ptrdiff_t n){
		m_index += n;
		m_iterator += n;
	}

	template<class I>
	std::ptrdiff_t distance_to(IndexedIterator<I> const& other) const{
		return other.m_iterator - m_iterator;
	}

	template<class I>
	bool equal(IndexedIterator<I> const& other) const{
		return m_iterator == other.m_iterator;
	}

	Iterator m_iterator;
	std::size_t m_index;
};

/// \brief Creates an iterator which reinterpretes an object as a range
///
/// The second template argument represents the elements by the proxy reference type. it must offer
/// a constructor Reference(sequence,i) which constructs a reference to the i-th proxy-element
template<class Sequence, class ValueType, class Reference>
class ProxyIterator: public SHARK_ITERATOR_FACADE<
	ProxyIterator<Sequence,ValueType,Reference>,
	ValueType,
	//boost::random_access_traversal_tag,
	std::random_access_iterator_tag,//keep VC quiet.
	Reference
>{
public:
	ProxyIterator() : m_position(0) {}

	ProxyIterator(Sequence& seq, std::size_t position)
	: m_sequence(&seq),m_position(position) {}

	template<class S, class V, class R>
	ProxyIterator(ProxyIterator<S,V,R> const& other)
	: m_sequence(other.m_sequence),m_position(other.m_position) {}

private:
	friend class SHARK_ITERATOR_CORE_ACCESS;
	template <class,class,class> friend class ProxyIterator;

	void increment() {
		++m_position;
	}
	void decrement() {
		--m_position;
	}

	void advance(std::ptrdiff_t n){
		m_position += n;
	}

	template<class Iter>
	std::ptrdiff_t distance_to(const Iter& other) const{
		return (std::ptrdiff_t)other.m_position - (std::ptrdiff_t)m_position;
	}

	template<class Iter>
	bool equal(Iter const& other) const{
		return (m_position == other.m_position);
	}
	Reference dereference() const {
		return Reference(*m_sequence,m_position);
	}

	Sequence* m_sequence;
	std::size_t m_position;
};

template<class Container>
class IndexingIterator: public SHARK_ITERATOR_FACADE<
	IndexingIterator<Container>,
	typename Container::value_type,
	std::random_access_iterator_tag,
	typename boost::mpl::if_<
		std::is_const<Container>,
		typename Container::const_reference,
		typename Container::reference
	>::type
>{
private:
	template<class> friend class IndexingIterator;
public:
	IndexingIterator(){}
	IndexingIterator(Container& container, std::size_t pos):m_container(&container), m_index(std::ptrdiff_t(pos)){}
	
	template<class C>
	IndexingIterator(IndexingIterator<C> const& iterator)
	: m_container(iterator.m_container){}

private:
	friend class SHARK_ITERATOR_CORE_ACCESS;

	void increment() {
		++m_index;
	}
	void decrement() {
		--m_index;
	}
	
	void advance(std::ptrdiff_t n){
		m_index += n;
	}
	
	template<class C>
	std::ptrdiff_t distance_to(IndexingIterator<C> const& other) const{
		return other.m_index - m_index;
	}
	
	template<class C>
	bool equal(IndexingIterator<C> const& other) const{
		return m_index == other.m_index;
	}
	typename IndexingIterator::reference dereference() const { 
		return (*m_container)[m_index];
	}
	
	Container* m_container;
	std::ptrdiff_t m_index;
};

}
#endif
