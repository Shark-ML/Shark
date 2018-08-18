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

template<class Container>
class IndexingIterator: public SHARK_ITERATOR_FACADE<
	IndexingIterator<Container>,
	typename Container::value_type,
	std::random_access_iterator_tag,
	typename std::conditional<
		std::is_const<Container>::value,
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
	: m_container(iterator.m_container), m_index(iterator.m_index){}

	std::size_t index() const{
		return (std::size_t) m_index;
	}
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
