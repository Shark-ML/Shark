/*!
 * 
 *
 * \brief       Implements one-point crossover operator.
 * 
 *
 * \author    T.Voss O.Krause
 * \date        2010-2011
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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

/// \brief Implements one-point crossover.
///
/// Given two input points of same size n, draws a random number between 0 and n-1. all variables
/// smaller than this index have the value of the left, all elements to the right have the value of the 
/// right parent.
struct OnePointCrossover {
	/// \brief Performs the one-point crossover
	template<typename PointType>
	PointType operator()( const PointType & mom, const PointType & dad ) {
		SIZE_CHECK(mom.size() == dad.size());
		std::size_t point = Rng::discrete( 0, mom.size() - 1 );
	    
		PointType offspring( mom.size() );
		std::copy( mom.begin(), mom.begin() + point, offspring.begin() );
		std::copy( dad.begin() + point, dad.end(), offspring.begin() + point );
	    
	    return offspring ;
		
	}
};
  
}

#endif
