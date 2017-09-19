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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_NSGA3_INDICATOR_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_NSGA3_INDICATOR_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/utility/KeyValuePair.h>
#include <shark/Algorithms/DirectSearch/Operators/Lattice.h>
#include <limits>
#include <vector>
#include <utility>

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
		
		//step 1: compute ideal point
		RealVector ideal = points.front();
		for(auto& point: points){
			noalias(ideal) = min(ideal,point);
		}
		//move current optimum in all objectives to 0
		for(auto& point: points)
			noalias(point) = point - ideal;
		RealVector normalizer = computeNormalizer(points);
		
		//step 2.3: create normalized fitness values
		for(auto& point: points)
			noalias(point) = point/ normalizer;
		
		typedef KeyValuePair<double,std::pair<std::size_t, std::size_t> > Pair;//stores (dist(p_j,z_i),j,i)
		// step 3: generate associative pairings between all points and the reference points
		std::vector<Pair> pairing(points.size(),makeKeyValuePair(std::numeric_limits<double>::max(),std::pair<std::size_t, std::size_t>()));
		for(std::size_t j = 0; j != points.size(); ++j){
			// find the reference this point is closest to
			for(std::size_t i = 0; i != m_Z.size(); ++i){
				//by pythagoras law we have a right triangle between our point x,
				//the projection onto the line z_i = <z_i,x>c_i and 0.
				//therefore we have |x-<z_i,x>z_i|^2 = |x|^2 - <z_i,x>^2
				// using |z_i| = 1
				double dist = norm_sqr(points[j]) - sqr(inner_prod(m_Z[i],points[j]));
				pairing[j] = min(pairing[j],makeKeyValuePair(dist,std::make_pair(j,i)));
			}
		}
		
		//check how points are assigned in the archive
		std::vector<std::size_t> rho(m_Z.size(),0);//rho_i counts the number of points assigned to Z_i
		std::size_t k = 0;
		for(; k != archive.size(); ++k){
			rho[pairing[k].value.second] ++;
		}
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
	
	template<typename Archive>
	void serialize( Archive & ar, const unsigned int ) {
		ar & m_Z;
	}
	
	void setReferencePoints(std::vector<RealVector> const& Z){
		m_Z = Z;
		
		for(auto& z: m_Z){
			z /= norm_2(z);
		}
	}
	
	template<class random>
	void init(std::size_t numOfObjectives, 
	          std::size_t mu, 
	          random& rng, 
	          std::vector<Preference> const & preferences = std::vector<Preference>()){
		std::size_t numLatticeTicks = computeOptimalLatticeTicks(numOfObjectives, mu);
		RealMatrix refs;
		if(preferences.empty())
		{
			refs = sampleLatticeUniformly(
				rng,
				unitVectorsOnLattice(numOfObjectives, numLatticeTicks),
				mu);
		}
		else
		{
			refs = preferenceAdjustedUnitVectors(
				numOfObjectives, numLatticeTicks, preferences);
		}
		// m_Z either has size mu or it, if preference points are given, its
		// size equals the number of preference points times the number of
		// points in the lattice structure.
		m_Z.resize(refs.size1());
		for(std::size_t i = 0; i < refs.size1(); ++i){
			m_Z[i] = row(refs, i);
		}
	}
private:
	std::vector<RealVector> m_Z;

	// approximates the points of the front by a plane spanned by the most extreme points.
	// then a normalizer is computed such that for the normalized points the plane has normal
	// (1,1,...,1). If this fails, the normalizer is chosen such that all values lie between 0 and 1.
	RealVector computeNormalizer(std::vector<RealVector> const& points)const{
		//step 1 find points spanning the plane
		double epsilon = 0.00001;
		std::size_t dimensions = points.front().size();
		RealMatrix cornerPoints(dimensions, dimensions,0.0);
		for(std::size_t dim = 0; dim != dimensions; ++dim){
			KeyValuePair<double,std::size_t> best(std::numeric_limits<double>::max(),0);
			for(std::size_t i = 0; i != points.size(); ++i){
				auto const& point = points[i];
				double dist = epsilon * sum(point) + (1-epsilon) * point[dim];
				best = std::min(best,makeKeyValuePair(dist,i));
			}
			noalias(row(cornerPoints,dim)) = points[best.value];
		}
		// compute plane equation
		// set up system of equations for linear regression
		// the plane equation is c_i^T w + b = 0 for all
		// corner points c_i. as w can be scaled freely, any value of b != 0
		// will give rise to a solution as long as the c_i are not linearly dependent.
		RealMatrix A = trans((cornerPoints|1)) % (cornerPoints|1);
		RealVector b = trans((cornerPoints|1)) % blas::repeat(-1.0,dimensions);//value of the right hand side of % does not matter as long as it is < 0
		blas::symm_pos_semi_definite_solver<RealMatrix> solver(A);
		if(solver.rank() == dimensions){//check if system is solvable
			solver.solve(b, blas::left());
			RealVector w = subrange(b,0,dimensions);// get the plane normal.
			if(min(w) >= 0){//check if linear factors make sense
				return blas::repeat(1.0,dimensions)/w;
			}
		}
		
		// if some of the error conditions are true,
		// we use the worst function values as
		// normalizer
		RealVector nadir = points.front();
		for(auto& point: points){
			noalias(nadir) = max(nadir,point);
		}
		return nadir;
		
		
	}
};

}

#endif
