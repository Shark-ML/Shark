/*!
 * 
 *
 * \brief       Implementation of the exact hypervolume calculation in 2 dimensions.
 *
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMECALCULATOR_2D_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMECALCULATOR_2D_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/utility/KeyValuePair.h>

#include <algorithm>
#include <vector>

namespace shark{
/// \brief Implementation of the exact hypervolume calculation in 2 dimensions.
///
///  The algorithm has complexity n log(n) and works by ordering the data by the first dimension
///  and then computing the upper riemann sum skipping dominated points
struct HypervolumeCalculator2D{

	/// \brief Executes the algorithm.
	/// \param [in] points The set \f$S\f$ of points for which to compute the volume
	/// \param [in] refPoint The reference point \f$\vec{r} \in \mathbb{R}^2\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$. .
	template<typename Set, typename VectorType >
	double operator()( Set const& points, VectorType const& refPoint){
		if(points.empty())
			return 0;
		SIZE_CHECK( points.begin()->size() == 2 );
		SIZE_CHECK( refPoint.size() == 2 );
		
		//copy set and order by first argument
		std::vector<shark::KeyValuePair<double,double> > set(points.size());
		for(std::size_t i = 0; i != points.size(); ++i){
			set[i].key = points[i][0];
			set[i].value = points[i][1];
		}
		std::sort( set.begin(), set.end());

		//go over the ordered set, skip dominated points and perform the integration
		double volume = ( refPoint[0] - set[0].key ) * ( refPoint[1] - set[0].value);

		std::size_t lastValidIndex = 0;
		for( std::size_t i = 1; i < set.size(); ++i ) {
			double diffDim1 = set[lastValidIndex].value - set[i].value;

			//diffDim1 <= 0 => point is dominated, so skip it
			if( diffDim1 > 0 ) {
				volume += ( refPoint[0] - set[i].key ) * diffDim1;
				lastValidIndex = i;
			}
		}
		return volume;
	}
};

}
#endif
