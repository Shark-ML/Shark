/*!
 * 
 * \file        EPTournamentSelection.h
 *
 * \brief       EP-Tournament selection operator.
 * 
 *
 * \author      T.Voss
 * \date        2010-2011
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_EP_TOURNAMENT_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_EP_TOURNAMENT_H

#include <shark/Algorithms/DirectSearch/EA.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/RouletteWheelSelection.h>

namespace shark {
    
    /** 
     * \brief Selects individuals from the range of parent and offspring individuals.
     */
    template< typename FitnessType >
    struct EPTournamentSelection {
        
        /** \brief Marks the user selected fitness type. */
        typedef FitnessType fitness_type;
        
        /** 
         * \brief Selects individuals from the range of parent and offspring individuals.         
         * \param [in] parents Iterator pointing to the first valid parent individual.
         * \param [in] parentsE Iterator pointing to the first invalid parent individual.
         * \param [in] offspring Iterator pointing to the first valid offspring individual.
         * \param [in] offspringE Iterator pointing to the first invalid offspring individual.
         * \param [in] out Iterator pointing to the first valid element of the output range.
         * \param [in] outE Iterator pointing to the first invalid element of the output range.
         * \param [in] tournamentSize Number of individuals participating in tournament.
         *
         */
        template<
        typename InIterator,  
            typename OutIterator
        > void operator()( InIterator parents, 
                          InIterator parentsE,
                          InIterator offspring, 
                          InIterator offspringE,
                          OutIterator out, 
                          OutIterator outE,
                          std::size_t tournamentSize ) {
            
            static FitnessComparator< FitnessType > comp;
            
            std::size_t noParents = std::distance( parents, parentsE );
            std::size_t noOffspring = std::distance( offspring, offspringE );
            std::size_t totalSize = noParents + noOffspring;
            
            std::vector< InIterator > view( totalSize );
            typename std::vector< InIterator >::iterator itv = view.begin();
            
            for( InIterator it = parents; it != parentsE; ++it, ++itv ) {
                it->fitness( shark::tag::ScaledFitness() )( 0 ) = 0.;
                *itv = it;
                for( std::size_t round = 0; round < tournamentSize; round++ ) {                    
                    std::size_t idx = shark::Rng::discrete( 0, totalSize-1 );                    
                    it->fitness( shark::tag::ScaledFitness() )( 0 ) += comp( *it, *( idx < noParents ? parents + idx : offspring + idx - noParents ) ) ? 1. : 0.;
                }
            }
            
            for( InIterator it = offspring; it != offspringE; ++it, ++itv ) {
                it->fitness( shark::tag::ScaledFitness() )( 0 ) = 0.;
                *itv = it;
                for( std::size_t round = 0; round < tournamentSize; round++ ) {                    
                    std::size_t idx = shark::Rng::discrete( 0, totalSize-1 );                    
                    it->fitness( shark::tag::ScaledFitness() )( 0 ) += comp( *it, *( idx < noParents ? parents + idx : offspring + idx - noParents ) ) ? 1. : 0.;
                }
            }
            
            std::sort( view.begin(), view.end(), shark::IndirectFitnessComparator< FitnessType >() );
            
            itv = view.begin();
            for( ; out != outE; ++out, ++itv )
                *out = **itv;
        }
        
        /** 
         * \brief Selects individuals from the range of parent and offspring individuals.         
         * \param [in] parents Range of parent individuals.
         * \param [in] offspring Range of offspring individuals.
         * \param [in] out Output range.
         * \param [in] tournamentSize Number of individuals participating in tournament.
         */
        template<
            typename InRange,  
            typename OutRange
        > void operator()( InRange parents, 
                          InRange offspring, 
                          OutRange out, 
                          std::size_t tournamentSize ) {
            (*this)( parents.begin(), parents.end(), offspring.begin(), offspring.end(), out.begin(), out.end(), tournamentSize );
        }
    };
    
}

#endif
