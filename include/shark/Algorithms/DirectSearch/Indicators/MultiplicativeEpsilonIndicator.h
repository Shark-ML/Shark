/*!
 *
 *
 * \brief       Calculates the multiplicate approximation quality of a Pareto-front
 * approximation.
 *
 *
 *
 * \author      T.Voss, O.Krause
 * \date        2010-2014
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_MULTIPLICATIVE_EPSILON_INDICATOR_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_MULTIPLICATIVE_EPSILON_INDICATOR_H

#include "AdditiveEpsilonIndicator.h"

#include <boost/range/adaptors/transformed.h>

namespace shark {

/// \brief Implements the Multiplicative approximation properties of sets
///
/// The multiplicative approximation measures by which factor a reference set has to be multiplied
/// before it becomes dominated by a target set.
///
/// The implemented least contributor algorithm calculates the point
/// That is approximated best by the remaining points. Thus the reference set is the full set and the target
/// sets all in which one point is removed.
///
/// Input points must be positive
///
/// See the following reference for further details:
///	- Bringmann, Friedrich, Neumann, Wagner. Approximation-Guided Evolutionary Multi-Objective Optimization. IJCAI '11.
struct MultiplicativeEpsilonIndicator {
	/// \brief Given a pareto front, returns the index of the point which is the least contributer
	template<typename ParetofrontType>
	unsigned int leastContributor(ParetofrontType const& front){
		typedef decltype(points[0]) Point;
		auto logTransform = [](Point const& x){return log(x);};
		AdditiveEpsilonIndicator indicator;
		return indicator.leastContributor(boost::adaptors::transform(points,logTransform));
	}
	
	/// \brief Updates the internal variables of the indicator using a whole population.
	///
	/// Empty for this Indicator
	template<typename ParetoFrontType>
	void updateInternals( ParetoFrontType const&){}
		
	template<typename Archive>
	void serialize( Archive &, const unsigned int ) {}
};

}

#endif
