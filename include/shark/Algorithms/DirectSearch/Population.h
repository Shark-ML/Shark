//===========================================================================
/*!
 * 
 *
 * \brief       Population on an Evolutionary Algorithm
 * 
 * 
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
//===========================================================================

#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_POPULATION_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_POPULATION_H

#include <shark/Algorithms/DirectSearch/EA.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/FitnessComparator.h>

#include <algorithm>
#include <vector>

namespace shark {


typedef std::vector< Individual > Population;

template<typename Population>
void shuffle( Population & p ) {
	std::random_shuffle( p.begin(), p.end() );
}

template<typename FitnessTag>
Population::const_iterator best_individual( const Population & p ) {
	return( std::min_element( p.begin(), p.end(), FitnessComparator<FitnessTag>() ) );
}

template<typename FitnessTag>
Population::const_iterator worst_individual( const Population & p ) {
	return( std::max_element( p.begin(), p.end(), FitnessComparator<FitnessTag>() ) );
}


}
#endif
