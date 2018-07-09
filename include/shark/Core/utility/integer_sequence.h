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
#ifndef SHARK_CORE_INTEGER_SEQUENCE_H
#define SHARK_CORE_INTEGER_SEQUENCE_H

#include <tuple>
namespace shark{ namespace detail{
	//taken from the web. implements an std::integer_sequence type representing a sequence 0,...,N-1, std::integer_sequence is not here until C++14
	template<int...> struct integer_sequence { using type = integer_sequence; };
	template<typename T1, typename T2> struct integer_sequence_concat;
	template<int... I1, int... I2> struct integer_sequence_concat<integer_sequence<I1...>, integer_sequence<I2...>>: integer_sequence<I1..., (sizeof...(I1) + I2)...> {};

	//generate_integer_sequence generates an integer sequence of integers 0....N. requires log(N) template instantiations
	template<int N> struct generate_integer_sequence;
	template<int N> struct generate_integer_sequence: integer_sequence_concat<typename generate_integer_sequence<N/2>::type, typename generate_integer_sequence<N-N/2>::type>::type {};
	template <> struct generate_integer_sequence<0>: integer_sequence<>{};
	template <> struct generate_integer_sequence<1>: integer_sequence<0>{};

}}
#endif
