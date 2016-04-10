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
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>

#include <algorithm>
#include <vector>

namespace shark {

///  \brief Calculates the hypervolume covered by a front of non-dominated points.
struct HypervolumeIndicator {
public:
	
	/// \brief Determines the point contributing the least hypervolume to the overall front of points.
	///
	/// This version uses the reference point estimated by the last call to updateInternals.
	///
	/// \param [in] front pareto front of points
	template<typename ParetoFrontType>
	std::size_t leastContributor( ParetoFrontType const& front)
	{
		std::vector<double> indicatorValues( front.size() );
		SHARK_PARALLEL_FOR( int i = 0; i < static_cast< int >( front.size() ); i++ ) {
			std::vector<RealVector> copy( front.begin(), front.end() );
			copy.erase( copy.begin() + i );
			
			HypervolumeCalculator hv;
			indicatorValues[i] = hv(copy,m_reference);
		}

		std::vector<double>::iterator it = std::max_element( indicatorValues.begin(), indicatorValues.end() );

		return it - indicatorValues.begin();
	}
	
	/// \brief Updates the internal variables of the indicator using a whole population.
	///
	/// Calculates the reference point of the volume from the population
	/// using the maximum value in every dimension+1
	///
	/// \param [in] front pareto front of points
	template<typename ParetoFrontType>
	void updateInternals( ParetoFrontType const& front){
		m_reference.clear();
		if(front.empty()) return;
		
		//calculate reference point
		std::size_t noObjectives = front[0].size();
		m_reference.resize(noObjectives);
		
		for(auto const& point: front)
			noalias(m_reference) = max(m_reference, point);
		
		noalias(m_reference) += 1.0;
	}

	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
		archive & BOOST_SERIALIZATION_NVP( m_reference );
	}

	RealVector m_reference;
};
}

#endif
