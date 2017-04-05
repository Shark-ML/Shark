/*!
 * 
 *
 * \brief       RealCodedNSGAIIII.h
 * 
 * 
 *
 * \author      O.Krause
 * \date        2017
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_REAL_CODED_NSGA_III_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_REAL_CODED_NSGA_III_H

// MOO specific stuff
#include <shark/Algorithms/DirectSearch/Operators/Indicators/NSGA3Indicator.h>
#include <shark/Algorithms/DirectSearch/RealCodedNSGAII.h>


namespace shark {

/// \brief Implements the NSGA-III
///
/// The NSGAIII works similar to the NSGAII except that the crowding distance is replaced by its own,
/// reference point based indicator.
///
/// Please see the following papers for further reference:
/// Deb, Kalyanmoy, and Himanshu Jain. 
/// "An evolutionary many-objective optimization algorithm using 
/// reference-point-based nondominated sorting approach, 
/// part I: Solving problems with box constraints."
/// IEEE Trans. Evolutionary Computation 18.4 (2014): 577-601.
class RealCodedNSGAIII : public IndicatorBasedRealCodedNSGAII<NSGA3Indicator>{		
public:

	/// \brief Default c'tor.
	RealCodedNSGAIII(random::rng_type& rng = random::globalRng):IndicatorBasedRealCodedNSGAII<NSGA3Indicator>(rng){}

	std::string name() const {
		return "RealCodedNSGAIII";
	}
};

}
#endif
