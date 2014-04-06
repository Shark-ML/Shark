/*!
 * 
 *
 * \brief       Provides a pair of Key and Value, as well as functions working with them.
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
#ifndef SHARK_CORE_KEY_VALUE_PAIR_H
#define SHARK_CORE_KEY_VALUE_PAIR_H

#include <shark/Core/utility/ZipPair.h>

#include <boost/operators.hpp>

#include <functional>//std::less
namespace shark{

///\brief Represents a Key-Value-Pair similar std::pair which is strictly ordered by it's key
///
///Key must be less-than comparable using operator<
template<class Key, class Value>
struct KeyValuePair
:boost::partially_ordered<KeyValuePair<Key,Value> >{
	Key key;
	Value value;
	
	KeyValuePair():key(), value(){}
	KeyValuePair(Key const& key, Value const& value)
	:key(key),value(value){}
	
	template<class Pair>
	KeyValuePair(Pair const& pair)
	:key(pair.key),value(pair.value){}
	
	template<class K, class V>
	bool operator==(KeyValuePair<K,V> const& pair) const{
		return key == pair.key;
	}
	template<class K, class V>
	bool operator<(KeyValuePair<K,V> const& pair) const{
		return key<pair.key;
	}
};

///\brief Swaps the contents of two instances of KeyValuePair
template<class K, class V>
void swap(KeyValuePair<K,V>& pair1, KeyValuePair<K,V>& pair2){
	using std::swap;
	swap(pair1.key,pair2.key);
	swap(pair1.value,pair2.value);
}

///\brief Creates a KeyValuePair
template<class Key, class Value>
KeyValuePair<Key,Value> makeKeyValuePair(Key const& key,Value const& value){
	return KeyValuePair<Key,Value>(key,value);
}


/// \cond

///\brief Reference type used by zipKeyValuePair
template<class Key, class Value,class KeyIterator, class ValueIterator>
struct PairReference<KeyValuePair<Key,Value>, KeyIterator, ValueIterator >{
private:
	typedef typename boost::iterator_reference<KeyIterator>::type KeyReference;
	typedef typename boost::iterator_reference<ValueIterator>::type ValueReference;
	typedef KeyValuePair<Key,Value> ReferedType;
public:
	struct type
	:boost::partially_ordered<type,ReferedType >{
		KeyReference key;
		ValueReference value;

		type(
			KeyReference key,
			ValueReference value
		):key(key),value(value){}

		template<class K, class V>
		type(
			KeyValuePair<K,V> const& pair
		):key(pair.key),value(pair.value){}

		template<class Reference>
		type& operator=(Reference const& pair){
			key = pair.key;
			value = pair.value;
			return *this;
		}
		type& operator=(type const& pair){
			key = pair.key;
			value = pair.value;
			return *this;
		}
	
		template<class T>
		bool operator==(T const& pair) const{
			return key == pair.key;
		}
		template<class T>
		bool operator<(T const& pair) const{
			return key < pair.key;
		}
	
		operator ReferedType()const{
			return ReferedType(key,value);
		}

		friend void swap(type a, type b){
			using std::swap;
			swap(a.key,b.key);
			swap(a.value,b.value);
		}
	};
};

/// \endcond

template<
	class Iterator1,
	class Iterator2
>
struct KeyValueRange: 
public boost::iterator_range<
	PairIterator<
		KeyValuePair<
			typename boost::iterator_value<Iterator1>::type,
			typename boost::iterator_value<Iterator2>::type
		>,
		Iterator1,Iterator2
	> 
>{
	typedef KeyValuePair<
			typename boost::iterator_value<Iterator1>::type,
			typename boost::iterator_value<Iterator2>::type
	> value_type; 
	typedef PairIterator<value_type,Iterator1,Iterator2> iterator;
	typedef boost::iterator_range<iterator> base_type;
	
	template<class Range1, class Range2>
	KeyValueRange(Range1& range1, Range2& range2)
	:base_type(zipPairRange<value_type>(range1, range2)){}
	
	KeyValueRange(Iterator1 begin1, Iterator1 end1, Iterator2 begin2, Iterator2 end2)
	:base_type(zipPairRange<value_type>(begin1, end1, begin2, end2)){}
	
	KeyValueRange(){}
};

///\brief Zips two ranges together, interpreting the first range as Key which can be sorted.
///
///\param begin1 beginning of first range
///\param end1 end of first range
///\param begin2 beginning of second range
///\param end2 end of second range
template<class Iterator1,class Iterator2>
KeyValueRange<Iterator1,Iterator2>
zipKeyValuePairs(Iterator1 begin1, Iterator1 end1, Iterator2 begin2, Iterator2 end2){
	return KeyValueRange<Iterator1,Iterator2>(begin1,end1,begin2,end2);
}


///\brief Zips two ranges together, interpreting the first range as Key which can be sorted.
///
///\param range1 The Key range
///\param range2 The value range
template<class Range1,class Range2>
KeyValueRange<
	typename boost::range_iterator<Range1>::type,
	typename boost::range_iterator<Range2>::type
>
zipKeyValuePairs(Range1& range1, Range2& range2){
	return KeyValueRange<
		typename boost::range_iterator<Range1>::type,
		typename boost::range_iterator<Range2>::type>(range1,range2);
}

}
#endif
