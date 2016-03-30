///
/// \brief       Swapper method for non-dominated sorting.
/// 
/// \author      T. Glasmachers
/// \date        2016
///
///
/// \par Copyright 1995-2016 Shark Development Team
/// 
/// <BR><HR>
/// This file is part of Shark.
/// <http://image.diku.dk/shark/>
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

#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_NONDOMINATEDSORT_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_NONDOMINATEDSORT_H

#include "FastNonDominatedSort.h"
#include "SweepingNonDominatedSort.h"


namespace shark {


///
/// \brief Wrapper for non-dominated sorting algorithms.
///
/// Assembles subsets/fronts of mutually non-dominated individuals.
/// Afterwards every individual is assigned a rank by pop[i].rank() = frontIndex.
/// The front of non-dominated points has the value 1.
///
template <class Extractor>
class BaseNonDominatedSort
{
	/// \brief Executes a non-dominated sorting algorithm.
	///
	/// Afterwards every individual is assigned a rank by pop[i].rank() = frontNumber.
	/// The front of dominating points has the value 1.
	///
	/// \param pop [in,out] Population to subdivide into fronts of non-dominated individuals.
	///
	template<typename PopulationType>
	void operator () (PopulationType& pop)
	{
		Extractor e;
		double n = pop.size();
		double m = e(pop[0]).size();
		// heuristic switching strategy based on simple benchmarks
		if (m == 2 || n > 5000 || log(n) / log(3.0) < m + 1.0)
		{
			BaseSweepingNonDominatedSort<Extractor> sorter;
			sorter(pop);
		}
		else
		{
			BaseFastNonDominatedSort<Extractor> sorter;
			sorter(pop);
		}
	}
};


/// \brief Non-dominated sorting based on the fitness.
typedef BaseNonDominatedSort< FitnessExtractor > NonDominatedSort;


};  // namespace shark
#endif
