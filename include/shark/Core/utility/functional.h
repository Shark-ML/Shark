/*!
 * 
 *
 * \brief       Small General algorithm collection.
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
#ifndef SHARK_CORE_FUNCTIONAL_H
#define SHARK_CORE_FUNCTIONAL_H

#include <boost/range/numeric.hpp>
#include <shark/Core/utility/Iterators.h>
#include <algorithm>
#include <shark/Core/Random.h>
namespace shark{
	
///\brief random_shuffle algorithm which stops after acquiring the random subsequence for [begin,middle)
template<class Iterator, class Rng>
void shuffle(Iterator begin, Iterator end, Rng& rng){
	using std::swap;
	Iterator next = begin;
	for (std::size_t index = 1; ++next != end; ++index){
	    swap(*next, *(begin + random::discrete(rng, std::size_t(0),index)));
	}
}

	
///\brief random_shuffle algorithm which stops after acquiring the random subsequence for [begin,middle)
template<class RandomAccessIterator, class Rng>
void partial_shuffle(RandomAccessIterator begin, RandomAccessIterator middle, RandomAccessIterator end, Rng& rng){
	shark::shuffle(begin,end,rng);
	// todo: test the algorithm below!
	//~ typedef typename std::iterator_traits<Iterator>::difference_type difference_type;
	//~ difference_type n = middle - begin;
	//~ for (; begin != middle; ++begin,--n) {
		
		//~ using std::swap;
		//~ swap(*begin, begin[rng(n)]);
	//~ }
}



///\brief random_shuffle algorithm which stops after acquiring the random subsequence for [begin,middle)
template<class RandomAccessIterator>
void partial_shuffle(RandomAccessIterator begin, RandomAccessIterator middle, RandomAccessIterator end){
	partial_shuffle(begin,middle,end,random::globalRng);
}

///\brief Returns the iterator to the median element. after this call, the range is partially ordered.
///
///After the call, all elements left of the median element are
///guaranteed to be <= median and all element on the right are >= median.
template<class Range>
typename boost::range_iterator<Range>::type median_element(Range& range){
	std::size_t size = range.size();
	std::size_t medianPos = (size+1)/2;
	auto medianIter = boost::begin(range)+medianPos;

	std::nth_element(range.begin(),medianIter, range.end());

	return medianIter;
}

template<class Range>
typename boost::range_iterator<Range>::type median_element(Range const& rangeAdaptor){
	Range adaptorCopy(rangeAdaptor);
	return median_element(adaptorCopy);
}
/// \brief Partitions a range in two parts as equal in size as possible.
///
/// The Algorithm partitions the range and returns the splitpoint. The elements in the range
/// are ordered such that all elements in [begin,splitpoint) < [splitpoint,end).
/// This partition is done such that the ranges are as equally sized as possible.
/// It is guaranteed that the left range is not empty. However, if the range consists only
/// of equal elements, the return value will be the end iterator indicating that there is no
/// split possible.
/// The whole algorithm runs in linear time by iterating 2 times over the sequence.
template<class Range>
typename boost::range_iterator<Range>::type partitionEqually(Range& range){
	auto begin = range.begin();
	auto end = range.end();
	auto medianIter = median_element(range);

	// in 99% of the cases we would be done right now. in the remaining 1% the median element is
	// not unique so we partition the left and the right such that all copies are ordered in the middle
	auto median = *medianIter;
	typedef typename Range::const_reference const_ref;
	auto left = std::partition(begin,medianIter,[&](const_ref elem){return elem < median;});
	auto right = std::partition(medianIter,end,[&](const_ref elem){return elem == median;});

	// we guarantee that the left range is not empty
	if(left == begin){
		return right;
	}

	// now we return left or right, maximizing size balance
	if (left - begin >= end - right)
		return left;
	else
		return right;
}

///\brief Partitions a range in two parts as equal in size as possible and returns it's result
///
///This the verison for adapted ranges.
template<class Range>
typename boost::range_iterator<Range>::type partitionEqually(Range const& rangeAdaptor){
	Range adaptorCopy(rangeAdaptor);
	return partitionEqually(adaptorCopy);
}

}
#endif
