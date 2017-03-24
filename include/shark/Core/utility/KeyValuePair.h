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
#ifndef SHARK_CORE_KEY_VALUE_PAIR_H
#define SHARK_CORE_KEY_VALUE_PAIR_H

namespace shark{

///\brief Represents a Key-Value-Pair similar std::pair which is strictly ordered by it's key
///
///Key must be less-than comparable using operator<
template<class Key, class Value>
struct KeyValuePair{
	Key key;
	Value value;
	
	KeyValuePair():key(), value(){}
	KeyValuePair(Key const& key, Value const& value)
	:key(key),value(value){}
	
	template<class K, class V>
	KeyValuePair(KeyValuePair<K,V> const& pair)
	:key(pair.key),value(pair.value){}
	
	template<class K, class V>
	bool operator==(KeyValuePair<K,V> const& pair) const{
		return key == pair.key;
	}
	template<class K, class V>
	bool operator!=(KeyValuePair<K,V> const& pair) const{
		return key != pair.key;
	}
	template<class K, class V>
	bool operator<(KeyValuePair<K,V> const& pair) const{
		return key < pair.key;
	}
	template<class K, class V>
	bool operator<=(KeyValuePair<K,V> const& pair) const{
		return key <= pair.key;
	}
	template<class K, class V>
	bool operator>(KeyValuePair<K,V> const& pair) const{
		return key > pair.key;
	}
	template<class K, class V>
	bool operator>=(KeyValuePair<K,V> const& pair) const{
		return key >= pair.key;
	}
	
	template<class Archive>
	void serialize(Archive &ar, const unsigned int /*file_version*/) {
		ar & key;
		ar & value;
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

/// \endcond

}
#endif
