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

	/**
	* \brief Tournament selection operator.
	*
	* See http://en.wikipedia.org/wiki/Fitness_proportionate_selection.
	*/
	struct TournamentSelection {
		/**
		* \brief Selects an individual from the range of individuals with prob. proportional to its fitness.
		* \tparam Extractor Type for mapping elements from the underlying range on the real line.
		* \param [in] it Iterator pointing to the first valid element.
		* \param [in] itE Iterator pointing to the first invalid element.
		* \param [in, out] e Object of type Extractor for mapping the elements from the underlying range on the real line.
		* \param [in] k Size of the tournament, needs to be larger than 0.
		* \returns An iterator pointing to the selected individual.
		*/
		template< typename Iterator, typename Extractor >
		Iterator operator()( Iterator it, Iterator itE, Extractor & e, std::size_t k ) const {

			if( k == 0 )
				throw( SHARKEXCEPTION( "TournamentSelection: Tournament size k needs to be larger than 0" ) );

			std::size_t n = std::distance( it, itE );

			if( n < k )
				throw( SHARKEXCEPTION( "TournamentSelecion: Size of population needs to be larger than size of tournament" ) );
			
			Iterator result = it + Rng::discrete( 0, n-1 );
			Iterator itt;
			for( std::size_t i = 1; i < k; i++ ) {
				itt = it + static_cast<unsigned int>( Rng::uni() * n );

				if( e( *itt ) < e( *result ) ) {
					std::swap( itt, result );
				}
			}

			return( result );
		}
	};

}

#endif