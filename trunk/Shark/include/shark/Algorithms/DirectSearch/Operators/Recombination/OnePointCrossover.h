/**
 *
 * \brief Implements one-point crossover operator.
 * \date 2010-2011
 * \par Copyright (c):
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 * <BR><HR>
 * This file is part of Shark. This library is free software;
 * you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software
 * Foundation; either version 3, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
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
