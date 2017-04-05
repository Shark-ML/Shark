/*!
 *
 *
 * \brief       Algorithm selecting points based on their crowding distance
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_CROWDING_DISTANCE_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_CROWDING_DISTANCE_H

#include <shark/LinAlg/Base.h>
#include <limits>

namespace shark {

/// \brief Implements the Crowding Distance of a pareto front
///
/// The Crowding distance is an estimate of the perimeter of the cuboid formed by
/// using the nearest neighbors of a given point as the vertices. 
/// The point with the smallest crowding distance is removed from the set.
///
/// See the following reference for further details:
/// Deb, Kalyanmoy, et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II."
///	IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
struct CrowdingDistance {
	/// \brief Selects the point with the smallest crowding distance
	///
	/// Crowding distance is computed wrt the union of the set front and the archive of points dominating the front
	template<typename ParetofrontType, typename ParetoArchive>
	std::size_t leastContributor(ParetofrontType const& front, ParetoArchive const& archive)const{
		if(front.size() < 2)
			return 0;
		std::size_t numDims = front[0].size();
		double keep = std::numeric_limits<double>::max();
		//compute the crowding distance
		std::vector<double> distances(front.size(),0.0);
		std::vector<KeyValuePair<double, unsigned int > > order(front.size() + archive.size());
		for( std::size_t i = 0; i != numDims; ++i ) {
			//create a joint set of front and archive
			for( std::size_t j = 0; j != front.size(); ++j ) {
				order[j].key = front[j][i];
				order[j].value = j;
			}
			for( std::size_t j = 0; j != archive.size(); ++j ) {
				order[j+front.size()].key = archive[j][i];
				order[j+front.size()].value = j+front.size();
			}
			//order to obtain neighbours
			std::sort(order.begin(),order.end());
			
			//check if we have to keep points because they are on the boundary
			if(order.front().value < front.size())
				distances[order.front().value] = keep;
			if(order.back().value < front.size())
				distances[order.back().value] = keep;
			double normalizer = order.back().key - order.front().key;
			for( std::size_t j = 1; j != order.size()-1; ++j ) {
				std::size_t index = order[j].value;
				//skip points which are part of the archive or which we want to keep
				if (index >= front.size() || distances[index] == keep)
					continue;
				
				//compute crowding distance
				distances[index] += (order[j+1].key - order[j-1].key)/normalizer;
			}
		}
		
		//get index of point with least crowding distance
		return std::min_element(distances.begin(), distances.end()) - distances.begin();
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
