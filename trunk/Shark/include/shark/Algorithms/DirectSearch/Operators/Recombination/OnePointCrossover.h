/*!
 * 
 *
 * \brief       Implements one-point crossover operator.
 * 
 *
 * \author      -
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_RECOMBINATION_ONE_POINT_CROSSOVER_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_RECOMBINATION_ONE_POINT_CROSSOVER_H

#include <shark/Rng/GlobalRng.h>

namespace shark {

    /**
     * \brief Implements one-point crossover.
     */
    struct TypedOnePointCrossover {
	
	template<typename ChromosomeType>
	ChromosomeType operator()( const ChromosomeType & mom, const ChromosomeType & dad, unsigned int point ) {
	    
	    if( mom.size() != dad.size() )
		throw( shark::Exception( "Parents need to be of the same size.", __FILE__, __LINE__ ) );
	    
	    ChromosomeType offspring( mom );
	    std::copy( dad.begin() + point, dad.end(), offspring.begin() + point );
	    
	    return( offspring );
	}

	template<typename ChromosomeType>
	ChromosomeType operator()( const ChromosomeType & mom, const ChromosomeType & dad ) {
	    return( (*this)( mom, dad, Rgn::discrete( 0, mom.size() - 1 ) ) );	    
	}

    }
  

}

#endif
