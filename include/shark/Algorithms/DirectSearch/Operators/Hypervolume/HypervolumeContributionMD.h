/*!
 *
 * \author      O.Krause, T. Glasmachers
 * \date        2014-2016
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUME_CONTRIBUTION_MD_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUME_CONTRIBUTION_MD_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/utility/KeyValuePair.h>
#include <shark/Core/Threading/Algorithms.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>
#include <shark/Algorithms/DirectSearch/Operators/Domination/NonDominatedSort.h>

#include <algorithm>
#include <vector>

namespace shark {
/// \brief Finds the hypervolume contribution for points in MD
///
/// This implementation is slightly less naive. Instead of calculating Con{x\in S}=Hyp{S}-Hyp{S/x}
/// directly, we restrict the volume dominated by points in S to be inside the box [x,ref]. This
/// leads to points in S not being relevant for the computation and thus can be discarded using
/// a simple dominance test.
struct HypervolumeContributionMD {
	/// \brief Returns the index of the points with smallest contribution.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the smallest contributor.
	/// \param [in] k The number of points to select.
	/// \param [in] referencePointThe reference Point\f$\vec{r} \in \mathbb{R}^2\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$.
	template<class Set, typename VectorType>
	std::vector<KeyValuePair<double,std::size_t> > smallest(Set const& points, std::size_t k, VectorType const& ref)const{
		SHARK_RUNTIME_CHECK(points.size() >= k, "There must be at least k points in the set");
		HypervolumeCalculator hv;
		std::vector<KeyValuePair<double,std::size_t> > result( points.size() );
		
		auto contribution = [&](std::size_t i){
			auto const& point = points[i];
			//compute restricted pointset
			std::vector<RealVector> pointset( points.begin(), points.end() );
			pointset.erase( pointset.begin() + i );
			restrictSet(pointset,point);
			
			double baseVol = std::exp(sum(log(ref-point)));
			result[i] ={baseVol - hv(pointset,ref), i};
		};
		threading::parallelND({points.size()}, {1}, contribution, threading::globalThreadPool());
		std::sort(result.begin(),result.end());
		result.erase(result.begin()+k,result.end());
		
		return result;
	}
	
	/// \brief Returns the index of the points with largest contribution.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the smallest contributor.
	/// \param [in] referencePointThe reference Point\f$\vec{r} \in \mathbb{R}^2\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$.
	template<class Set, typename VectorType>
	std::vector<KeyValuePair<double,std::size_t> > largest(Set const& points, std::size_t k, VectorType const& ref)const{
		SHARK_RUNTIME_CHECK(points.size() >= k, "There must be at least k points in the set");
		HypervolumeCalculator hv;
		std::vector<KeyValuePair<double,std::size_t> > result( points.size() );
		
		auto contribution = [&](std::size_t i){
			auto const& point = points[i];
			//compute restricted pointset
			std::vector<RealVector> pointset( points.begin(), points.end() );
			pointset.erase( pointset.begin() + i );
			restrictSet(pointset,point);
			
			double baseVol = std::exp(sum(log(ref-point)));
			result[i] ={baseVol - hv(pointset,ref), i};
		};
		threading::parallelND({points.size()}, {1}, contribution, threading::globalThreadPool());
		std::sort(result.begin(),result.end());
		result.erase(result.begin(),result.end()-k);
		std::reverse(result.begin(),result.end());
		return result;
	}
	
	/// \brief Returns the index of the points with smallest contribution.
	///
	/// As no reference point is given, the extremum points can not be computed and are never selected.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the smallest contributor.
	/// \param [in] k The number of points to select.
	template<class Set>
	std::vector<KeyValuePair<double,std::size_t> > smallest(Set const& points, std::size_t k)const{
		SHARK_RUNTIME_CHECK(points.size() >= k, "There must be at least k points in the set");
		//find reference point as well as points with lowest function value
		std::vector<std::size_t> minIndex(points[0].size(),0);
		RealVector minVal = points[0];
		RealVector ref=points[0];
		for(std::size_t i = 1; i != points.size(); ++i){
			noalias(ref) = max(ref,points[i]);
			for(std::size_t j = 0; j != minVal.size(); ++j){
				if(points[i](j)< minVal[j]){
					minVal[j] = points[i](j);
					minIndex[j]=i;
				}
			}
		}
		// compute the k smallest elements with the computed reference points.
		// add a few points extra so that we can ensure that we can filter out the corner points
		auto resultsAll = smallest(points, std::min(points.size(), k+minIndex.size()), ref);
		std::vector<KeyValuePair<double,std::size_t> > pruned;
		for( auto const& elem: resultsAll){
			if(std::find(minIndex.begin(),minIndex.end(),elem.value) == minIndex.end())
				pruned.push_back(elem);
			if(pruned.size() == k) break;
		}
		return pruned;
	}
	
	/// \brief Returns the index of the points with largest contribution.
	///
	/// As no reference point is given, the extremum points can not be computed and are never selected.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the smallest contributor.
	/// \param [in] k The number of points to select.
	template<class Set>
	std::vector<KeyValuePair<double,std::size_t> > largest(Set const& points, std::size_t k)const{
		SHARK_RUNTIME_CHECK(points.size() >= k, "There must be at least k points in the set");
		//find reference point as well as points with lowest function value
		std::vector<std::size_t> minIndex(points[0].size(),0);
		RealVector minVal = points[0];
		RealVector ref=points[0];
		for(std::size_t i = 1; i != points.size(); ++i){
			noalias(ref) = max(ref,points[i]);
			for(std::size_t j = 0; j != minVal.size(); ++j){
				if(points[i](j)< minVal[j]){
					minVal[j] = points[i](j);
					minIndex[j]=i;
				}
			}
		}
		
		//compute the k largest elements with the computed reference points.
		// add a few points extra so that we can ensure that we can filter out the corner points
		auto resultsAll = largest(points, std::min(points.size(), k+minIndex.size()), ref);
		std::vector<KeyValuePair<double,std::size_t> > pruned;
		for( auto const& elem: resultsAll){
			if(std::find(minIndex.begin(),minIndex.end(),elem.value) == minIndex.end())
				pruned.push_back(elem);
			if(pruned.size() == k) break;
		}
		return pruned;
	}
private:
	/// \brief Restrict the points to the area covered by point and remove all points which are then dominated
	template<class Pointset, class Point>
	void restrictSet(Pointset& pointset, Point const& point) const{
		for(auto& p: pointset){
			noalias(p) = max(p,point);
		}
		std::vector<std::size_t> ranks(pointset.size());
		nonDominatedSort(pointset,ranks);
		std::size_t end = pointset.size();
		std::size_t pos = 0;
		while(pos != end){
			if(ranks[pos] == 1){
				++pos;
				continue;
			}
			--end;
			if(pos != end){
				std::swap(pointset[pos],pointset[end]);
				std::swap(ranks[pos],ranks[end]);
			}
		}
		pointset.erase(pointset.begin() +end, pointset.end());
	}
};

}
#endif
