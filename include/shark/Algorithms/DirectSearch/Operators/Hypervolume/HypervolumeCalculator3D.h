/*!
 * 
 *
 * \brief       Implementation of the exact hypervolume calculation in 3 dimensions.
 *
 *
 * \author      T. Glasmachers
 * \date        2016
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMECALCULATOR_3D_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMECALCULATOR_3D_H

#include <shark/LinAlg/Base.h>

#include <algorithm>
#include <vector>
#include <map>

namespace shark {
/// \brief Implementation of the exact hypervolume calculation in 3 dimensions.
///
/// M. T. M. Emmerich and C. M. Fonseca.
/// Computing hypervolume contributions in low dimensions: Asymptotically optimal algorithm and complexity results.
/// In: Evolutionary Multi-Criterion Optimization (EMO) 2011.
/// Vol. 6576 of Lecture Notes in Computer Science, pp. 121--135, Berlin: Springer, 2011.
struct HypervolumeCalculator3D {
	/// \brief Executes the algorithm.
	/// \param [in] points The set of points for which to compute the volume
	/// \param [in] refPoint The reference point \f$\vec{r} \in \mathbb{R}^3\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$.
	template<typename Set, typename VectorType >
	double operator()( Set const& points, VectorType const& refPoint)
	{
		if (points.empty()) return 0;
		SIZE_CHECK(points.begin()->size() == 3);
		SIZE_CHECK(refPoint.size() == 3);

		std::vector<VectorType> set;
		set.resize(points.size());
		for (std::size_t i=0; i<points.size(); i++) set[i] = points[i];
		std::sort(set.begin(), set.end(),
					[] (VectorType const& x, VectorType const& y)
					{ return (x[2] < y[2]); }
//~					{
//~						if (x[2] < y[2]) return true;
//~						if (x[2] > y[2]) return false;
//~						if (x[0] < y[0]) return true;
//~						if (x[0] > y[0]) return false;
//~						return (x[1] < y[1]);
//~					}
				);

		double volume = 0.0;
		double area = 0.0;
		std::map<double, double> front2D;
		double prev_x2 = 0.0;
		for (size_t i=0; i<set.size(); i++)
		{
			VectorType const& x = set[i];
			if (i > 0) volume += area * (x[2] - prev_x2);

			// check whether x is dominated
			std::map<double, double>::iterator worse = front2D.upper_bound(x[0]);
			double b = refPoint[1];
			if (worse != front2D.begin())
			{
				std::map<double, double>::iterator better = worse;
				if (better == front2D.end() || better->first > x[0]) --better;
				if (better->second <= x[1]) continue;
				b = better->second;
			}

			// remove dominated points
			while (worse != front2D.end())
			{
				if (worse->second < x[1]) break;
				std::map<double, double>::iterator it = worse;
				++worse;
				double r = (worse == front2D.end()) ? refPoint[0]: worse->first;
				area -= (r - it->first) * (b - it->second);
				front2D.erase(it);
			}

			// insert x
			front2D[x[0]] = x[1];
			double r = (worse == front2D.end()) ? refPoint[0] : worse->first;
			area += (r - x[0]) * (b - x[1]);

			prev_x2 = x[2];
		}
		volume += area * (refPoint[2] - prev_x2);
		return volume;
	}
};

}
#endif
