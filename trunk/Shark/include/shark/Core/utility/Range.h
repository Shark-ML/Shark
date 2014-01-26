/*!
 * 
 *
 * \brief       Helper Methods to use with boost range.
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
#ifndef SHARK_CORE_RANGE_H
#define SHARK_CORE_RANGE_H
#include <boost/range.hpp>
#include <utility>
#include <shark/Core/Exception.h>

namespace shark{
///\brief returns the size of a range
///
/// This is just a fix to a bug of boost::size which produces a lot of useless warnings
template<class Range>
std::size_t size(Range const& range){
	return (std::size_t)boost::size(range); 
}
///\brief returns the i-th element of a range
template<class Range>
typename boost::range_reference<Range>::type
get( Range& range, std::size_t i){
	SIZE_CHECK(i < shark::size(range));
	typename boost::range_iterator<Range>::type pos=boost::begin(range);
	std::advance(pos,i);
	return *pos;
}
template<class Range>
typename boost::range_reference<Range const>::type
get( Range const& range, std::size_t i){
	SIZE_CHECK(i < shark::size(range));
	typename boost::range_iterator<Range const>::type pos=boost::begin(range);
	std::advance(pos,i);
	return *pos;
}


}
#endif
