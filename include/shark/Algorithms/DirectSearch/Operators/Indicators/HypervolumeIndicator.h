/*!
 * 
 *
 * \brief       Calculates the hypervolume covered by a front of non-dominated points.
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_HYPERVOLUMEINDICATOR_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_HYPERVOLUMEINDICATOR_H

#include <shark/Core/Exception.h>
#include <shark/Core/OpenMP.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeContribution.h>

#include <algorithm>
#include <vector>

namespace shark {

///  \brief Calculates the hypervolume covered by a front of non-dominated points.
///
/// If given, the Indicator uses a provided reference value that can be set via setReference. 
/// Otherwise, it is computed from the data by using the maximum value in the set. As this usually
/// gives 0 contribution to the extremum points (i.e. the ones with best function value), those
/// points are skipped when computing the contribution (i.e. extremum points are never selected).
/// Note, that for boundary points that are not extrema, this does not hold and they are selected.
///
/// for problems with many objectives, an approximative algorithm can be used.
struct HypervolumeIndicator {
	/// \brief Determines the point contributing the least hypervolume to the overall front of points.
	///
	/// \param [in] front pareto front of points
	template<typename ParetoFrontType>
	std::size_t leastContributor( ParetoFrontType const& front)const{
		HypervolumeContribution algorithm;
		if(m_reference.size() != 0)
			return m_algorithm.smallest(front,1,m_reference)[0].value;
		else	
			return m_algorithm.smallest(front,1)[0].value;
	}
	
	/// \brief Sets the reference point. 
	///
	/// If no point is set, it is estimated from the current front and the extremum points are never selected.
	void setReference(RealVector const& newReference){
		m_reference = newReference;
	}
	
	/// \brief Whether the approximtive algorithm should be used on large problems
	void useApproximation(bool useApproximation){
		m_algorithm.useApproximation(useApproximation);
	}
	
	///\brief Error bound for the approximative algorithm
	double approximationEpsilon()const{
		return m_algorithm.approximationEpsilon();
	}
	///\brief Error bound for the approximative algorithm
	double& approximationEpsilon(){
		return m_algorithm.approximationEpsilon();
	}
	
	///\brief Error probability for the approximative algorithm
	double approximationDelta()const{
		return m_algorithm.approximationDelta();
	}
	
	///\brief Error probability for the approximative algorithm
	double& approximationDelta(){
		return m_algorithm.approximationDelta();
	}

	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
		archive & BOOST_SERIALIZATION_NVP( m_reference );
		archive & BOOST_SERIALIZATION_NVP( m_algorithm );
	}

private:
	RealVector m_reference;
	HypervolumeContribution m_algorithm;
};
}

#endif
