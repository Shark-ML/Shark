/*!
 *
 * \author      O.Krause, T. Glasmachers
 * \date        2014-2016
 *
 *
 * \par Copyright 1995-2016 Shark Development Team
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUME_CONTRIBUTION_MD_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUME_CONTRIBUTION_MD_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/utility/KeyValuePair.h>
#include <shark/Core/OpenMP.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>

#include <algorithm>
#include <vector>

namespace shark {
/// \brief Finds the hypervolume contribution for points in MD
///
/// This is the naive default algorithm explicitely calculating Con{x\in S}=Hyp{S}-Hyp{S/x} for all x in the set of points S
struct HypervolumeContributionMD {
	/// \brief Returns the index of the points with smallest contribution.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the smallest contributor.
	/// \param [in] k The number of points to select.
	/// \param [in] referencePointThe reference Point\f$\vec{r} \in \mathbb{R}^2\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$.
	template<class Set, typename VectorType>
	std::vector<KeyValuePair<double,std::size_t> > smallest(Set const& points, std::size_t k, VectorType const& ref)const{
		HypervolumeCalculator hv;
		double baseVol = hv(points,ref);
		std::vector<KeyValuePair<double,std::size_t> > result( points.size() );
		SHARK_PARALLEL_FOR( int i = 0; i < static_cast< int >( points.size() ); i++ ) {
			std::vector<RealVector> copy( points.begin(), points.end() );
			copy.erase( copy.begin() + i );
			
			result[i].key = baseVol-hv(copy,ref);
			result[i].value = i;
		}
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
		HypervolumeCalculator hv;
		double baseVol = hv(points,ref);
		std::vector<KeyValuePair<double,std::size_t> > result( points.size() );
		SHARK_PARALLEL_FOR( int i = 0; i < static_cast< int >( points.size() ); i++ ) {
			std::vector<RealVector> copy( points.begin(), points.end() );
			copy.erase( copy.begin() + i );
			
			HypervolumeCalculator hv;
			result[i].key = baseVol-hv(copy,ref);
			result[i].value = i;
		}
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
		HypervolumeCalculator hv;
		double baseVol = hv(points,ref);
		std::vector<KeyValuePair<double,std::size_t> > result;
		SHARK_PARALLEL_FOR( int i = 0; i < static_cast< int >( points.size() ); i++ ) {
			if(std::find(minIndex.begin(),minIndex.end(),i) != minIndex.end())
				continue;
			std::vector<RealVector> copy( points.begin(), points.end() );
			copy.erase( copy.begin() + i );
			
			double volume = baseVol - hv(copy,ref);
			SHARK_CRITICAL_REGION{
				result.emplace_back(volume,i);
			}
		}
		std::sort(result.begin(),result.end());
		result.erase(result.begin()+k,result.end());
		
		return result;
	}

	
	
	
	
	/// \brief Returns the index of the points with largest contribution.
	///
	/// As no reference point is given, the extremum points can not be computed and are never selected.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the smallest contributor.
	/// \param [in] k The number of points to select.
	template<class Set>
	std::vector<KeyValuePair<double,std::size_t> > largest(Set const& points, std::size_t k)const{
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
		
		HypervolumeCalculator hv;
		double baseVol = hv(points,ref);
		std::vector<KeyValuePair<double,std::size_t> > result;
		SHARK_PARALLEL_FOR( int i = 0; i < static_cast< int >( points.size() ); i++ ) {
			if(std::find(minIndex.begin(),minIndex.end(),i) != minIndex.end())
				continue;
			std::vector<RealVector> copy( points.begin(), points.end() );
			copy.erase( copy.begin() + i );
			
			HypervolumeCalculator hv;
			double volume = baseVol - hv(copy,ref);
			SHARK_CRITICAL_REGION{
				result.emplace_back(volume,i);
			}
		}
		std::sort(result.begin(),result.end());
		result.erase(result.begin(),result.end()-k);
		std::reverse(result.begin(),result.end());
		return result;
	}
};

}
#endif