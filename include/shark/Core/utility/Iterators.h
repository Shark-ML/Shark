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
#ifndef SHARK_CORE_ITERATORS_H
#define SHARK_CORE_ITERATORS_H

#include <boost/version.hpp>
#include "Impl/boost_iterator_facade_fixed.hpp"//thanks, boost.

#include <shark/Core/utility/Range.h>
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
class ProxyIterator: public  SHARK_ITERATOR_FACADE<
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


namespace detail{

	///\brief Helper class of the MultiSequenceIterator, which querys everything needed to deduce the right iterator_facade
	template<class Self, class SequenceContainer>
	struct SequenceOfSequenceIteratorTraits{
		typedef typename boost::remove_const<SequenceContainer>::type OuterSequence;
		typedef typename boost::range_iterator<SequenceContainer>::type outer_iterator;
		//the inner Sequence type is the value_type of the outer sequence. But if the outer sequence is const
		//we have to make sure, that we also get const value_type.
		//~ typedef typename CopyConst<typename boost::iterator_value<outer_iterator>::type,SequenceContainer>::type InnerSequence;
		typedef typename boost::remove_reference<
			typename boost::iterator_reference<outer_iterator>::type
		>::type InnerSequence;
		typedef typename boost::range_iterator<InnerSequence>::type inner_iterator;
		typedef typename boost::iterator_reference<inner_iterator>::type reference;
		typedef typename boost::iterator_value<inner_iterator>::type value_type;

		//typedef boost::iterator_facade_fixed< Self, value_type, boost::random_access_traversal_tag, reference > base;
		typedef  SHARK_ITERATOR_FACADE< Self, value_type, std::random_access_iterator_tag, reference > base;
	};
}

///\brief Iterator which iterates of the elements of a nested sequence
///
///Think about a sequence which is split in several parts. These parts
///are than stored into a new sequence. An example for this is std::deque
///or the Data class. This iterator let's you tierate over the elements of the sequence
///without having to care about that the sequence itself is splitted.
///The Sequences both have to be random access containers.
template<class SequenceContainer>
class MultiSequenceIterator: public detail::SequenceOfSequenceIteratorTraits<
	MultiSequenceIterator<SequenceContainer>,
	SequenceContainer
>::base{
private:
	typedef detail::SequenceOfSequenceIteratorTraits<
		MultiSequenceIterator<SequenceContainer>,SequenceContainer
	> Traits;
	typedef typename Traits::outer_iterator outer_iterator;
public:
	typedef typename Traits::inner_iterator inner_iterator;
	MultiSequenceIterator()
	:m_positionInSequence(0){}

	template<class OuterIter, class InnerIter>
	MultiSequenceIterator(
		OuterIter outerBegin,
		OuterIter outerEnd,
		OuterIter outerPosition,
		InnerIter innerPosition,
		std::size_t positionInSequence
	):m_outerBegin(outer_iterator(outerBegin)),
	m_outerPosition(outer_iterator(outerPosition)),
	m_outerEnd(outer_iterator(outerEnd)),
	m_innerPosition(innerPosition),
	m_positionInSequence(positionInSequence){
		//we can't dereference if we are past the end...
		if(m_outerPosition != m_outerEnd){
			m_innerBegin = boost::begin(*m_outerPosition);
			m_innerEnd = boost::end(*m_outerPosition);
		}

	}


	template<class S>
	MultiSequenceIterator(MultiSequenceIterator<S> const& other)
	: m_outerBegin(other.m_outerBegin),m_outerPosition(other.m_outerPosition),m_outerEnd(other.m_outerEnd),
	m_innerBegin(other.m_innerBegin),m_innerPosition(other.m_innerPosition),m_innerEnd(other.m_innerEnd),
	m_positionInSequence(other.m_positionInSequence) {}
		
	std::size_t index()const{
		return m_positionInSequence;
	}
	
	inner_iterator getInnerIterator()const{
		return m_innerPosition;
	}

private:
	friend class SHARK_ITERATOR_CORE_ACCESS;
	template <class> friend class MultiSequenceIterator;

	void increment() {
		++m_positionInSequence;
		++m_innerPosition;
		if(m_innerPosition == m_innerEnd){
			++m_outerPosition;
			while (m_outerPosition != m_outerEnd){//support for empty subsequences
				m_innerBegin = boost::begin(*m_outerPosition);
				m_innerEnd = boost::end(*m_outerPosition);
				if(m_innerBegin != m_innerEnd){
					m_innerPosition = m_innerBegin;
					return;
				}
				++m_outerPosition;
			}
		}
	}
	void decrement() {
		SIZE_CHECK(m_positionInSequence);//don't call this method when the iterator is on the first element
		--m_positionInSequence;
		if(m_innerPosition != m_innerBegin){
			--m_innerPosition;
			return;
		}
			
		--m_outerPosition;
		m_innerBegin = boost::begin(*m_outerPosition);
		m_innerEnd = boost::end(*m_outerPosition);
		while(m_innerBegin == m_innerEnd){//support for empty subsequences
			if( m_outerPosition == m_outerBegin)
				return;//we are at the end
			--m_outerPosition;
			m_innerBegin = boost::begin(*m_outerPosition);
			m_innerEnd = boost::end(*m_outerPosition);
		}
		m_innerPosition = m_innerEnd-1;
	}
	//this is not exactly O(1) as the standard wants. in fact it's O(n) in the number of inner sequences
	//so approximately O(1) if the size of a sequence is big...
	void advance(std::ptrdiff_t n){
		m_positionInSequence += n;
		std::ptrdiff_t diff = m_innerPosition - m_innerBegin;
		n += diff;//jump from the start of the current inner sequence
		if(n== 0)
			m_innerPosition = m_innerBegin;
		if(n < 0){
			n *= -1;
			--m_outerPosition;
			--n;
			//jump over the outer position until we are in the correct range again
			while ((unsigned int) n >= shark::size(*m_outerPosition) ){
				n -= shark::size(*m_outerPosition);
				--m_outerPosition;
			}
			//get the iterators to the current position if we are not before the beginning of the sequence
			m_innerBegin = boost::begin(*m_outerPosition);
			m_innerEnd = boost::end(*m_outerPosition);
			m_innerPosition = m_innerEnd-(n+1);
		}
		else{

			//jump over the outer position until we are in the correct range again
			while (m_outerPosition != m_outerEnd && (unsigned int)n >= shark::size(*m_outerPosition) ){
				n -= shark::size(*m_outerPosition);
				++m_outerPosition;
			}
			SHARK_CHECK(m_outerPosition != m_outerEnd || (n == 0), "iterator went past the end");
			//get the iterators to the current position if we are not past the end
			if(m_outerPosition != m_outerEnd){
				m_innerBegin = boost::begin(*m_outerPosition);
				m_innerPosition = m_innerBegin+n;
				m_innerEnd = boost::end(*m_outerPosition);
			}
		}
	}

	template<class Iter>
	std::ptrdiff_t distance_to(const Iter& other) const{
		return (std::ptrdiff_t)other.m_positionInSequence - (std::ptrdiff_t)m_positionInSequence;
	}

	template<class Iter>
	bool equal(Iter const& other) const{
		return (m_positionInSequence == other.m_positionInSequence);
	}
	typename Traits::reference dereference() const {
		return *m_innerPosition;
	}

	outer_iterator m_outerBegin;//in fact, it is a before-the-begin iterator
	outer_iterator m_outerPosition;
	outer_iterator m_outerEnd;

	inner_iterator m_innerBegin;//in fact, it is a before-the-begin iterator
	inner_iterator m_innerPosition;
	inner_iterator m_innerEnd;

	std::size_t m_positionInSequence;
};

}
#endif
