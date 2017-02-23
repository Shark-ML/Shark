/*!
 * 
 *
 * \brief       Implementation of the exact hypervolume calculation in 3 dimensions.
 *
 *
 * \author      T. Glasmachers
 * \date        2016-2017
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
	double operator()( Set const& points, VectorType const& refPoint){
		if (points.empty()) return 0.0;
		SIZE_CHECK(points.begin()->size() == 3);
		SIZE_CHECK(refPoint.size() == 3);

		std::vector<VectorType> set;
		for (std::size_t i=0; i<points.size(); i++)
		{
			VectorType const& p = points[i];
			if (p[0] < refPoint[0] && p[1] < refPoint[1] && p[2] < refPoint[2]) set.push_back(p);
		}
		if (set.empty()) return 0.0;
		std::sort(set.begin(), set.end(),
					[] (VectorType const& x, VectorType const& y)
					{ return (x[2] < y[2]); }
				);

		// add the first point
		std::map<double, double> front2D;
		VectorType const& x0 = set[0];
		front2D[x0[0]] = x0[1];
		double prev_x2 = x0[2];
		double area = (refPoint[0] - x0[0]) * (refPoint[1] - x0[1]);
		double volume = 0.0;

		// process further points
		for (size_t i=1; i<set.size(); i++)
		{
			assert(! front2D.empty());
			assert(area > 0.0);

			VectorType const& x = set[i];

			// check whether x is dominated and find "top" coordinate
			double t = refPoint[1];
			std::map<double, double>::iterator right = front2D.lower_bound(x[0]);
			std::map<double, double>::iterator left = right;
			if (right == front2D.end())
			{
				--left;
				t = left->second;
			}
			else
			{
				if (right->first == x[0])
				{
					t = left->second;
				}
				else if (left != front2D.begin())
				{
					--left;
					t = left->second;
				}
			}
			if (x[1] >= t) continue;   // x is dominated

			// add chunk to volume
			volume += area * (x[2] - prev_x2);

			// remove dominated points and corresponding areas
			while (right != front2D.end() && right->second >= x[1])
			{
				std::map<double, double>::iterator tmp = right;
				++right;
				const double r = (right == front2D.end()) ? refPoint[0] : right->first;
				area -= (r - tmp->first) * (t - tmp->second);
				front2D.erase(tmp);
			}

			// add the new point
			const double r = (right == front2D.end()) ? refPoint[0] : right->first;
			area += (r - x[0]) * (t - x[1]);
			front2D[x[0]] = x[1];

			// volume is processed up to here:
			prev_x2 = x[2];
		}

		// add trailing chunk to volume
		volume += area * (refPoint[2] - prev_x2);

		// return the result
		return volume;
	}
};

}
#endif
