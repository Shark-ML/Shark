/*!
 * \brief       Implements the fast non-dominated sort algorithm.
 * 
 * \author      T. Voss, O. Krause
 * \date        2015
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_DOMINATION_FASTNONDOMINATEDSORT_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_DOMINATION_FASTNONDOMINATEDSORT_H

#include <shark/Algorithms/DirectSearch/Operators/Domination/ParetoDominance.h>
#include <vector>

namespace shark {


/// \brief Implements the well-known non-dominated sorting algorithm.
///
/// Assembles subsets/fronts of mututally non-dominating individuals.
/// Afterwards every individual is assigned a rank by points[i].rank() = frontNumber.
/// The front of dominating points has the value 1. 
///
/// The algorithm is dscribed in Deb et al, 
/// A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II
/// IEEE Transactions on Evolutionary Computation, 2002
/// \param points [in] pointsulation to subdivide into fronts of non-dominated individuals.
/// \param ranks [out] Set of integers storing the rank of the i-th point. must have the same size
template<class PointRange, class RankRange>
void fastNonDominatedSort(PointRange const& points, RankRange& ranks) {
	SIZE_CHECK(points.size() == ranks.size());
	//stores for the i-th point which points are dominated by i
	std::vector<std::vector<std::size_t> > s(points.size());
	//stores for every point how many points are dominating it
	std::vector<std::size_t> numberOfDominatingPoints(points.size(), 0);
	//stores initially the front of non-dominated points
	std::vector<std::size_t> front;
	for (std::size_t i = 0; i < points.size(); i++) {
		//check which points j are dominated by i and add them to s[i]
		//also increment n[j] for every i dominating j
		for (std::size_t  j = 0; j < points.size(); j++) {
			if (i == j) continue;
			DominanceRelation rel = dominance(points[i], points[j]);
			if (rel == LHS_DOMINATES_RHS)
				s[i].push_back(j);
			else if (rel == RHS_DOMINATES_LHS)
				numberOfDominatingPoints[i]++;
		}
		//all non-dominated points form the first front
		if (numberOfDominatingPoints[i] == 0){
			front.push_back(i);
			ranks[i] = 1;//non-dominated points have rank 1
		}
	}
	//find subsequent fronts.
	unsigned frontCounter = 2;
	std::vector<std::size_t> nextFront;

	//as long as we can find fronts
	//visit all points of the last front found and remove them from the
	//set. All points which are not dominated anymore form the next front
	while (!front.empty()) {
		//visit all points of the current front and remove them
		// if any point is not dominated, it is part the next front.
		for(std::size_t element = 0; element != front.size(); ++element) {
			//visit all points dominated by the element
			std::vector<std::size_t> const& dominatedPoints = s[front[element]];
			for (std::size_t  j = 0; j != dominatedPoints.size(); ++j){
				std::size_t point = dominatedPoints[j];
				numberOfDominatingPoints[point]--;
				// if no more points are dominating this, add to the next front.
				if (numberOfDominatingPoints[point] == 0){
					nextFront.push_back(point);
					ranks[point] = frontCounter;
				}
			}
		}

		// make the newly found front the current one
		front.swap(nextFront);
		nextFront.clear();
		frontCounter++;
	}
}

}//namespace shark
#endif
