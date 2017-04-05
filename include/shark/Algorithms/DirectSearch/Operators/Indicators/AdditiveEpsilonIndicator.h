/*!
 *
 *
 * \brief       Calculates the additive approximation quality of a Pareto-front
 * approximation.
 *
 *
 *
 * \author      T.Voss, O.Krause
 * \date        2010-2014
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_ADDITIVE_EPSILON_INDICATOR_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_ADDITIVE_EPSILON_INDICATOR_H

#include <shark/LinAlg/Base.h>
#include <limits>

namespace shark {

/// \brief Implements the Additive approximation properties of sets
///
/// The additive approximation measures which value must be subtracted from a reference set
/// until it becomes dominated by a target set.
///
/// The implemented least contributor algorithm calculates the point
/// That is approximated best by the remaining points. Thus the reference set is the full set and the target
/// sets all in which one point is removed.
///
/// See the following reference for further details:
///	- Bringmann, Friedrich, Neumann, Wagner. Approximation-Guided Evolutionary Multi-Objective Optimization. IJCAI '11.
struct AdditiveEpsilonIndicator {
	/// \brief Given a pareto front, returns the index of the point which is the least contributer
	///
	/// The archive has no effect on the volume as the archive is dominating all points in the front
	template<typename ParetoFrontType, typename ParetoArchive>
	std::size_t leastContributor( ParetoFrontType const& front, ParetoArchive const& /*archive*/)const{
		std::size_t leastIndex = 0;
		double leastValue = std::numeric_limits<double>::max();
		for( std::size_t i = 0; i != front.size(); i++ ) {
			//find the minimum distance the front with one point removed has to be moved to dominate the original front
			double result = std::numeric_limits<double>::max();
			for(std::size_t j = 0; j != front.size(); ++j){
				if(j == i) continue; //this point is removed
				result = std::min(result,max(front[j]-front[i])); 
			}
			if(result < leastValue){
				leastValue = result;
				leastIndex = i;
			}
		}
		//~ std::cout<<leastIndex<<" "<<leastValue<<std::endl;
		return leastIndex;
	}
	
	template<typename ParetoFrontType, typename ParetoArchive>
	std::vector<std::size_t> leastContributors( ParetoFrontType const& front, ParetoArchive const& archive, std::size_t K)const{
		std::vector<std::size_t> indices;
		std::vector<RealVector> points(front.begin(),front.end());
		std::vector<std::size_t> activeIndices(points.size());
		std::iota(activeIndices.begin(),activeIndices.end(),0);
		for(std::size_t k=0; k != K; ++k){
			std::size_t index = leastContributor(points,archive);
			
			points.erase(points.begin()+index);
			indices.push_back(activeIndices[index]);
			activeIndices.erase(activeIndices.begin()+index);
		}
		return indices;
	}
	
	template<class random>
	void init(std::size_t /*numOfObjectives*/, std::size_t /*mu*/, random& /*rng*/){}
		
	template<typename Archive>
	void serialize( Archive &, const unsigned int ) {}
};

}

#endif
