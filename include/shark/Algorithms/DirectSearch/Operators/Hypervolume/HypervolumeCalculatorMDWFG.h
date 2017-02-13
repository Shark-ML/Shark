/*!
 * 
 *
 * \brief       Implementation of the exact hypervolume calculation in m dimensions.
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMECALCULATOR_MD_WFG_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMECALCULATOR_MD_WFG_H

#include <shark/LinAlg/Base.h>
#include <shark/Algorithms/DirectSearch/Operators/Domination/NonDominatedSort.h>
#include <algorithm>
#include <vector>
#include <map>

namespace shark {
/// \brief Implementation of the exact hypervolume calculation in m dimensions.
///
///  The algorithm is described in
///
/// L. While, L. Bradstreet and L. Barone, "A Fast Way of Calculating Exact Hypervolumes," 
/// in IEEE Transactions on Evolutionary Computation, vol. 16, no. 1, pp. 86-95, Feb. 2012.
///
/// WFG is extremely fast in practice, while theoretically it has O(2^N) complexity where N
/// is the number of points.
///
/// We do not implement slicing as the paper showed that it does have only small impact
/// while it increases the algorithm complexity dramatically.
struct HypervolumeCalculatorMDWFG {

	/// \brief Executes the algorithm.
	/// \param [in] set The set \f$S\f$ of points for which the following assumption needs to hold: \f$\forall s \in S: \lnot \exists s' \in S: s' \preceq s \f$
	/// \param [in] refPoint The reference point \f$\vec{r} \in \mathbb{R}^n\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$. .
	template<typename Set, typename VectorType >
	double operator()( Set const& points, VectorType const& refPoint)const{
		if(points.empty())
			return 0;
		SIZE_CHECK( points.begin()->size() == refPoint.size() );
		
		std::vector<VectorType> set(points.begin(),points.end());
		std::sort( set.begin(), set.end(), [ ](VectorType const& x, VectorType const& y){return x.front() > y.front();});
		return wfg(set,refPoint);
	}
	
private:
	
	template<class Set, class VectorType>
	double wfg(Set const& points, VectorType const& refPoint)const{
		//first handle a few special cases as they are likely faster to compute than the recursion
		std::size_t n = points.size(); 
		if(n == 0){
			return 0;
		}
		if(n == 1){
			return boxVolume(points[0],refPoint);
		}
		if(n == 2){
			double volume1 = boxVolume(points[0],refPoint);
			double volume2 = boxVolume(points[1],refPoint);
			double volume3 = boxVolume(max(points[0], points[1]),refPoint);
			return volume1 + volume2 - volume3;
		}
		//by default we recurse and compute the sum of hypercontributions
		//using Hyp(S) = sum_i HypCon{x_i|(x1,...,x_{i-1}}
		//and S={x1...x_N}. 
		//We can next make use of the fact that HypCon can restrict
		//the dominated set of S to the set dominated by x_i. This
		//allows us to throw points away which do not affect the volume.
		//This makes the algorithm fast as we can hope to throw away
		//points quickly in early iterations.
		double volume = 0;
		for(std::size_t i = 0; i != points.size(); ++i){
			auto const& point = points[i];
			//compute restricted pointset wrt point
			std::vector<RealVector> pointset( points.begin()+i+1, points.end() );
			limitSet(pointset,point);
			
			double baseVol = boxVolume(point,refPoint);
			volume += baseVol - wfg(pointset,refPoint);
		}
		return volume;
	}

	/// \brief Restrict the points to the area covered by point and remove all points which are then dominated
	template<class Pointset, class Point>
	void limitSet(Pointset& pointset, Point const& point) const{
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
		std::sort( pointset.begin(), pointset.end(), [ ](Point const& x, Point const& y){return x.front() > y.front();});
	
	}
	template<class Point1, class Point2>
	double boxVolume(Point1 const& p,Point2 const& ref) const {
		double volume = 1;
		for(std::size_t i = 0; i != ref.size(); ++i){
			volume *= ref(i) - p(i);
		}
		return volume;
	}
};
}
#endif
