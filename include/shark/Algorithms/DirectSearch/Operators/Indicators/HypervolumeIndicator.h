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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_HYPERVOLUMEINDICATOR_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_HYPERVOLUMEINDICATOR_H

#include <shark/Core/Exception.h>
#include <shark/Core/OpenMP.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeContribution.h>

#include <algorithm>
#include <vector>
#include <numeric>

namespace shark {

///  \brief Selects the least contributing points based on hypervolume contribution.
///
/// For problems with many objectives, an approximative algorithm for estimating the contribution can be used.
///
/// The hypervolume indicator requires a reference point describing the worst possible function value, which can be set via setReference. 
/// In many cases, this point is hard to define as error functions are often unbounded.
/// In this case, the reference point can be estimated from the data (as a lower bound of the true reference point).
/// There are two ways this is done depending on whether an archive of points is supplied or not.
/// 1. The supplied archive is empty. In this case, the reference point is estimated purely from the supplied set as the maximum in each objective.
/// As this usually gives 0 contribution to the extremum points (i.e. the ones with best function value), those
/// points are skipped when computing the contribution (i.e. extremum points are never selected).
/// Note, that for boundary points that are not extrema, this does not hold and they can still be selected
/// Warning: this entails that the set must have at least numObjectives+K points!
///
/// 2. An archive is provided with additional points. In this case we assume the supplied front is either
/// dominated or extremum points are explicitely handled. Thus the reference point is computed as the maximum of
/// the union of the points in set and archive and all points can be selected.
struct HypervolumeIndicator {
	/// \brief Determines the point contributing the least hypervolume to the overall front of points.
	///
	/// \param [in] front pareto front of points
	/// \param [in] archive Optional archive of external points used for estimate the reference point if the reference point is not set
	template<typename ParetoFrontType, typename ParetoArchive>
	std::size_t leastContributor( ParetoFrontType const& front, ParetoArchive const& archive)const{
		HypervolumeContribution algorithm;
		if(m_reference.size() != 0)
			return m_algorithm.smallest(front,1,m_reference)[0].value;
		else if(archive.empty())//no reference point, case 1
			return m_algorithm.smallest(front,1)[0].value;
		else{//no reference point, case 2
			auto ref = front[0];                        
			for(auto const& p: front){
				noalias(ref) = max(ref,p);
			}
			for(auto const& p: archive){
				noalias(ref) = max(ref,p);
			}
			return m_algorithm.smallest(front,1, ref)[0].value;
		}
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
