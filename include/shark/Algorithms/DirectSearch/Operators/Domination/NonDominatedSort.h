///
/// \brief       Swapper method for non-dominated sorting.
/// 
/// \author      T. Glasmachers
/// \date        2016
///
///
/// \par Copyright 1995-2017 Shark Development Team
/// 
/// <BR><HR>
/// This file is part of Shark.
/// <http://shark-ml.org/>
/// 
/// Shark is free software: you can redistribute it and/or modify
/// it under the terms of the GNU Lesser General Public License as published 
/// by the Free Software Foundation, either version 3 of the License, or
/// (at your option) any later version.
/// 
/// Shark is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU Lesser General Public License for more details.
/// 
/// You should have received a copy of the GNU Lesser General Public License
/// along with Shark.  If not, see <http://www.gnu.org/licenses/>.
///

#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_DOMINATION_NONDOMINATEDSORT_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_DOMINATION_NONDOMINATEDSORT_H

#include "FastNonDominatedSort.h"
#include "DCNonDominatedSort.h"


namespace shark {

/// \brief Frontend for non-dominated sorting algorithms.
///
/// Assembles subsets/fronts of mutually non-dominated individuals.
/// Afterwards every individual is assigned a rank by pop[i].rank() = frontIndex.
/// The front of non-dominated points has the value 1.
///
/// Depending on dimensionality m and number of points n, either the 
/// fastNonDominatedSort algorithm with O(n^2 m) or the dcNonDominatedSort
/// alforithm with complexity O(n log(n)^m) is called.
template<class PointRange, class RankRange>
void nonDominatedSort(PointRange const& points, RankRange& ranks) {
	SIZE_CHECK(points.size() == ranks.size());
	std::size_t n = points.size();
	if(n == 0) return;
	std::size_t m = points[0].size();
	// heuristic switching strategy based on simple benchmarks
	if (m == 2 || n > 5000 || std::log(n) / log(3.0) < m + 1.0)
	{
		dcNonDominatedSort(points,ranks);
	}
	else
	{
		fastNonDominatedSort(points,ranks);
	}
}

//version that takes temporary ranges as second argument.
//this allows nonDominatedSort(points,ranks(population) as the second argument will return a temporary proxy
//we would like to use r-value references here but gcc 4.8 appears to be buggy in that regard
template<class PointRange, class RankRange>
//~ void nonDominatedSort(PointRange const& points, RankRange&& ranks) {
void nonDominatedSort(PointRange const& points, RankRange const& ranks) {
	RankRange ranksCopy=ranks;
	nonDominatedSort(points,ranksCopy);
}


} // namespace shark
#endif
