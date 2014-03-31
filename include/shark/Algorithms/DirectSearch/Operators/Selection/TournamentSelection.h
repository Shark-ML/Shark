/*!
 *
 *
 * \brief       Implements tournament selection.
 *
 * See http://en.wikipedia.org/wiki/Tournament_selection
 *
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_TOURNAMENT_SELECTION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_TOURNAMENT_SELECTION_H

#include <shark/Core/Exception.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {

/// \brief Tournament selection operator.
///
/// Selects k individuals at random and returns the best. By default k is 2.
/// The Template parameter represents a Predicate that compares two individuals. It is assumed that this is
/// transitive, that is pred(a,b) = true && pred(b,c) == true => pred(a,c) == true.
/// Possible predicates could compare fitness values or the rank of the two individuals.
///
/// The size of the tournament can either be set in the constructor or by setting the variable tournamentSize
template<class Predicate>
struct TournamentSelection {
	TournamentSelection(std::size_t size = 2){
		tournamentSize = 2;
	}
	
	template<typename IteratorType1, typename IteratorType2>
	void operator()(
		IteratorType1 inIt,
		IteratorType1 inItE,
		IteratorType2 outIt,
		IteratorType2 outItE
	){
		for(; outIt != outItE; ++outIt ) {
			*outIt = (*this)(inIt,inItE);
		}
	}
	
	/// \brief Selects an individual from the range of individuals with prob. proportional to its fitness.
	/// \param [in] it Iterator pointing to the first valid element.
	/// \param [in] itE Iterator pointing to the first invalid element.
	/// \return An iterator pointing to the selected individual.
	template< typename Iterator>
	Iterator operator()( Iterator it, Iterator itE) const
	{
		std::size_t n = std::distance( it, itE );
		SHARK_CHECK(tournamentSize > 0, " Tournament size k needs to be larger than 0");
		SHARK_CHECK(n > tournamentSize, " Size of population needs to be larger than size of tournament");
		
		Predicate predicate;
		Iterator result = it + Rng::discrete( 0, n-1 );
		for( std::size_t i = 1; i < tournamentSize; i++ ) {
			Iterator itt = it + Rng::discrete(0,n-1);
			if( predicate(*itt, *result) ){
				result = itt;
			}
		}

		return result;
	}
	
	/// \brief Size of the tournament. 2 by default.
	std::size_t tournamentSize;
};

}

#endif