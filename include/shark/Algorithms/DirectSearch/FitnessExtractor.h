/*!
 * 
 *
 * \brief       Explicit traits for extracting fitness values from arbitrary types
 * 
 * 
 *
 * \author      T.Voss, O.Krause
 * \date        2010-2014
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_FITNESS_EXTRACTOR_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_FITNESS_EXTRACTOR_H

#include <shark/LinAlg/Base.h>

namespace shark {

/**
* \brief Functor that returns its argument without conversion
*/
struct IdentityFitnessExtractor {
	template<typename Member>
	const Member & operator()( const Member & member ) const {
		return member;	
	}

};

struct FitnessExtractor {
	template<typename Individual>
		typename Individual::FitnessType const& operator()( Individual const& individual ) const {
		return individual.penalizedFitness();
	}
};
}

#endif
