/*!
 *
 *
 * \brief       Algorithm selecting front based on their crowding distance
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_NSGAIII_SELECTION_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_NSGAIII_SELECTION_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/utility/KeyValuePair.h>
#include <limits>

namespace shark {


struct NSGA3Indicator {
	template<typename ParetofrontType, typename ParetoArchive>
	std::vector<std::size_t> leastContributors(ParetofrontType const& front, ParetoArchive const& archive, std::size_t K)const{
		//copy both fronts together into temporary.
		//remember: archived(preselected) points are at the front, points to select are at the back
		std::vector<RealVector> points;
		for(auto const& point: archive)
			points.push_back(point);
		for(auto const& point: front)
			points.push_back(point);
		
		//step 1: compute ideal and nadir over all points
		RealVector ideal = points[0];
		RealVector nadir = points[0];
		for(auto& point: points){
			noalias(ideal) = min(ideal,point);
			noalias(nadir) = max(nadir,point);
		}
		//step 2 compute normalizer
		//todo: by default the algorithm uses a weird way to compute the hyper plane
		//it can sometimes fail and then we would need to fallback to the below anyways
		//~ //step 2: find front closest mapping to the objective axis
		//~ double epsilon = 0.00001;
		//~ std::vector<std::size_t> z_max(ideal.size(),0);
		//~ std::vector<double> z_dist(ideal.size(),std::numeric_limits<double>::max());
		//~ for(std::size_t dim = 0; dim != ideal.size(); ++dim){
			//~ for(std::size_t i = 0; i != front.size(); ++i){
				//~ auto const& point = front[i];
				//~ double dist = epsilon * sum(point) + (1-epsilon) * point[j];
				//~ if(dist< z_dist[dim]){
					//~ dist = z_dist[dim];
					//~ z_max[dim] = i;
				//~ }
			//~ }
		//~ }
		//~ //step 2.2: find the equation of the plane spanned
		//~ RealMatrix Z(ideal.size(),ideal.size());
		RealVector normalizer = nadir - ideal;
		
		//step 2.3: create normalized fitness values
		for(auto& point: archive)
			noalias(point) = (point - ideal) / normalizer;
		
		
		typedef KeyValuePair<double,std::pair<std::size_t, std::size_t> > Pair;//stores (dist(p_j,z_i),j,i)
		// step 3: generate associative pairings between all points and the reference points
		std::vector<Pair> pairing(points.size(),makeKeyValuePair(std::numeric_limits<double>::max(),std::pair<std::size_t, std::size_t>()));
		for(std::size_t i = 0; i != Z.size(); ++i){
			double norm = norm_sqr(Z[i]);
			auto projection = 1 - outer_prod(Z[i],Z[i])/ norm;//projection to the plane with normal Z_i
			for(std::size_t j = 0; j != points.size(); ++j){
				double dist = norm_sqr(prod(projection,points[j]));
				pairing[j] = min(pairing[j],makeKeyValuePair(dist,std::make_pair(j,i)));
			}
		}
		
		
		//check how points are assigned in the archive
		std::vector<std::size_t> rho(Z.size(),0);
		std::size_t k = 0;
		for(; k != archive.size(); ++k){
			rho[pairing[k].value.second] ++;
		}
		//~ std::cout<<k<<std::endl;
		// step 4: select the points to keep
		while(k < points.size() - K){
			//find reference point that got least points assigned;
			std::size_t index = std::min_element(rho.begin(),rho.end()) - rho.begin();
			//Search for an unassigned associated point that is close to it
			bool found = false;
			KeyValuePair<double,std::size_t > closest(std::numeric_limits<double>::max(),0);
			for(std::size_t i = k; i != points.size(); ++i){
				if(pairing[i].value.second == index){//is this point associated?
					found = true;
					closest = std::min(closest,makeKeyValuePair(pairing[i].key,i));
				}
			}
			//~ std::cout<<found<<" "<<closest.value<<" "<<pairing[closest.value].value.first<<std::endl;
			//if no point was found, we disregard the reference point from now on
			if(!found){
				rho[index] = points.size() +1;
			}else{
				//we remove it from the list of unassigned points
				swap(pairing[k],pairing[closest.value]);
				//we found a point and assign it
				rho[index]++;
				++k;
			}
		}
		
		//return the indices of the remaining unselected points
		std::vector<std::size_t> unselected;
		for(; k != points.size(); ++k){
			SIZE_CHECK(pairing[k].value.first >= archive.size());
			unselected.push_back(pairing[k].value.first - archive.size());
		}
		return unselected;
		
	}
	
	std::vector<RealVector> Z;
		
	template<typename Archive>
	void serialize( Archive &, const unsigned int ) {}
};

}

#endif
