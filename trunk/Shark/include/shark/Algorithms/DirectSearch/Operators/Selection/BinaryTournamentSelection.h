/*!
 *
 *
 * \brief       Implements binary tournament selection.
 *
 * See http://en.wikipedia.org/wiki/Tournament_selection
 *
 * \par Copyright (c) 1998-2008:
 * Institut f&uuml;r Neuroinformatik
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_BINARY_TOURNAMENT_SELECTION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_BINARY_TOURNAMENT_SELECTION_H

#include <shark/Rng/GlobalRng.h>

namespace shark {

/**
* \brief Implements binary tournament selection with a user-selectable predicate.
* \tparam Predicate Predicate type for comparing candidate solutions.
*/
template<typename Predicate>
struct BinaryTournamentSelection {

	/**
	* \brief C'tor.
	* \param [in,out] predicate Predicate instance for comparing candidate solutions.
	*/
	BinaryTournamentSelection( Predicate & predicate ) : m_predicate( predicate )
	{
	}

	/**
	* \brief Selects individuals from range [inIt,inItE) and places them in
	* the range [outIt, outItE).
	*/
	template<typename IteratorType1, typename IteratorType2>
	void operator()( IteratorType1 inIt,
	                 IteratorType1 inItE,
	                 IteratorType2 outIt,
	                 IteratorType2 outItE
	               )
	{
		std::size_t inSize = std::distance( inIt, inItE );
		std::size_t i1, i2;
		while( outIt != outItE ) {

			i1 = Rng::discrete( 0, inSize - 1);
			i2 = Rng::discrete( 0, inSize - 1);

			if( m_predicate( *(inIt + i2), *(inIt + i1) ) )
				*outIt = *(inIt + i2);
			else
				*outIt = *(inIt + i1);

			++outIt;
		}
	}

	/**
	* \brief Selects and returns an individual from range [inIt,inItE).
	*/
	template<typename IteratorType>
	IteratorType operator()( IteratorType inIt, IteratorType inItE )
	{
		std::size_t inSize = std::distance( inIt, inItE );
		std::size_t i1, i2;

		i1 = Rng::discrete( 0, inSize - 1);
		i2 = Rng::discrete( 0, inSize - 1);

		return( m_predicate( *(inIt + i2), *(inIt + i1) ) ? inIt + i2 : inIt + i1 );
	}

	Predicate & m_predicate; ///< Predicate instance.
};
}

#endif