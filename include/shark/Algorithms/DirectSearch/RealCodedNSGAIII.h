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
/// \ingroup multidirect
class RealCodedNSGAIII : public IndicatorBasedRealCodedNSGAII<NSGA3Indicator>{
	typedef IndicatorBasedRealCodedNSGAII<NSGA3Indicator> base;
public:

	/// \brief Default c'tor.
	RealCodedNSGAIII(random::rng_type& rng = random::globalRng)
		: base(rng){}

	std::string name() const {
		return "RealCodedNSGAIII";
	}
protected:
	void doInit(
		std::vector<SearchPointType> const& initialSearchPoints,
		std::vector<ResultType> const& functionValues,
		RealVector const& lowerBounds,
		RealVector const& upperBounds,
		std::size_t mu,
		double nm,
		double nc,
		double crossover_prob,
		std::vector<Preference> const & indicatorPreferences = std::vector<Preference>()
		){
		// Do the regular initialization.
		base::doInit(
			initialSearchPoints, 
			functionValues,
			lowerBounds,
			upperBounds,
			mu,
			nm,
			nc,
			crossover_prob);
		// Make sure that the indicator respects our preference points if they
		// are set.
		indicator().init(functionValues.front().size(), mu, *mpe_rng, 
		                 indicatorPreferences);
	}
};

}
#endif
