/*!
 * 
 *
 * \brief       Range which zips two ranges together while allowing a custom pair type
 * 
 * 
 * 
 *
 * \author      Oswin Krause
 * \date        2012
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
#ifndef SHARK_CORE_ZIPPAIR_H
#define SHARK_CORE_ZIPPAIR_H

#include <shark/Core/utility/Range.h>
#ifdef SHARK_USE_ITERATOR_WORKAROUND
#include "Impl/boost_iterator_facade_fixed.hpp"//thanks, boost.
#define SHARK_ITERATOR_FACADE boost::iterator_facade_fixed
#define SHARK_ITERATOR_CORE_ACCESS boost::iterator_core_access_fixed
#else
#define SHARK_ITERATOR_FACADE boost::iterator_facade
#define SHARK_ITERATOR_CORE_ACCESS boost::iterator_core_access
#endif
#include <utility>

namespace shark{

///\brief Given a type of pair and two iterators to zip together, returns the reference
///
///e.g. for std::pair<T,U> the reference is std::pair<T&,U&>
template<class Pair, class Iterator1, class Iterator2>
struct PairReference;

/// \cond

template<class T, class U,class Iterator1, class Iterator2>
struct PairReference<std::pair<T, U>, Iterator1, Iterator2 >{
	typedef std::pair<
		typename boost::iterator_reference<Iterator1>::type,
		typename boost::iterator_reference<Iterator2>::type
	> type;
};

/// \endcond

///\brief A Pair-Iterator which gives a unified view of two ranges.
///
/// A Pair-Iterator is an iterator which zips two ranges together and returns the zipped result.
/// This implementation allows the return type to be an arbirtrary Pair which is then constructed using
/// Reference(*iter1,*iter2) when the iterator is dereferenced. This allows for more expressive element access
/// instead of iter->first or iter->second.
/// \todo Both underlying Iterators must be random access iterators.
//todo: implement as decorator for boost::zip_iterator
template<class Value,class Iterator1,class Iterator2>
class PairIterator: public SHARK_ITERATOR_FACADE<
	PairIterator<Value,Iterator1,Iterator2>,
	Value,
	std::random_access_iterator_tag,
	typename PairReference<Value,Iterator1,Iterator2>::type
>{
private:
	typedef typename PairReference<Value,Iterator1,Iterator2>::type Reference;
	template<class,class,class> friend class PairIterator;
public:
	PairIterator(){}
	
	///\brief Copy-Constructs this iterator from some other IndexedIterator convertible to this.
	template< class IteratorT1,class IteratorT2>
	PairIterator(IteratorT1 const& iterator1, IteratorT2 const& iterator2)
	: m_iterator1(iterator1),m_iterator2(iterator2){}
	
	template<class V,class I1,class I2>
	PairIterator(PairIterator<V,I1,I2> const& iterator)
	: m_iterator1(iterator.m_iterator1),m_iterator2(iterator.m_iterator2){}
	
	Iterator1 first()const{
		return m_iterator1;
	}
	
	Iterator2 second()const{
		return m_iterator2;
	}
	
	std::pair<Iterator1,Iterator2> iterators()const{
		return std::make_pair(m_iterator1,m_iterator2);
	}

private:
	friend class SHARK_ITERATOR_CORE_ACCESS;

	void increment() {
		++m_iterator1;
		++m_iterator2;
	}
	void decrement() {
		--m_iterator1;
		--m_iterator2;
	}
	
	void advance(std::ptrdiff_t n){
		m_iterator1 += n;
		m_iterator2 += n;
	}
	
	template<class V,class I1,class I2>
	std::ptrdiff_t distance_to(PairIterator<V,I1,I2> const& other) const{
		return  other.m_iterator1 - m_iterator1;
	}
	
	template<class V,class I1,class I2>
	bool equal(PairIterator<V,I1,I2> const& other) const{
		return m_iterator1 == other.m_iterator1;
	}
	Reference dereference() const { 
		return Reference(*m_iterator1,*m_iterator2);
	}
	
	Iterator1 m_iterator1;
	Iterator2 m_iterator2;
};

///\brief returns a paired zip range using pair type Pair
///This class must be specialized for every Pair to be used
template<class PairType, class Iterator1, class Iterator2>
boost::iterator_range<
	PairIterator<PairType,Iterator1,Iterator2>
>
zipPairRange(Iterator1 begin1, Iterator1 end1, Iterator2 begin2,Iterator2 end2){
	typedef PairIterator<PairType,Iterator1,Iterator2> iterator;
	return make_iterator_range(iterator(begin1,begin2),iterator(end1,end2));
}

template<class PairType, class Range1, class Range2>
struct PairRangeType{
	typedef boost::iterator_range<
		PairIterator<
			PairType,
			typename boost::range_iterator<Range1>::type,
			typename boost::range_iterator<Range2>::type
		>
	> type;
};

///\brief returns a paired zip range using pair type Pair
///
/// usage: zipPairRange<PairType>(range1,range2) leads to a range consisting
/// where the i-th element is the Pair PairType(range1[i],range2[i]).
template<class PairType, class Range1, class Range2>
typename PairRangeType<PairType, Range1, Range2>::type
zipPairRange(Range1 & range1, Range2& range2){
	return zipPairRange<PairType>(boost::begin(range1), boost::end(range1), boost::begin(range2),boost::end(range2));
}



///\brief returns a paired zip range using pair type Pair
///
/// usage: zipPairRange<PairType>(range1,range2) leads to a range consisting
/// where the i-th element is the Pair PairType(range1[i],range2[i]).
template<class PairType, class Range1, class Range2>
typename PairRangeType<PairType, Range1 const, Range2 const>::type
zipPairRange(Range1 const& range1, Range2 const& range2){
	return zipPairRange<PairType>(boost::begin(range1), boost::end(range1), boost::begin(range2),boost::end(range2));
}

}
#endif
