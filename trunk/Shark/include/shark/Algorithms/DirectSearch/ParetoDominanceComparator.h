/*!
 * 
 *
 * \brief       Implementation of the Pareto-Dominance relation.
 * 
 * The function is described in
 * 
 * Christian Igel, Nikolaus Hansen, and Stefan Roth. 
 * Covariance Matrix Adaptation for Multi-objective Optimization. 
 * Evolutionary Computation 15(1), pp. 1-28, 2007
 * 
 * 
 *
 * \author      T.Voss
 * \date        2011-2014
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_PARETODOMINANCECOMPARATOR_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_PARETODOMINANCECOMPARATOR_H

#include <shark/Core/Exception.h>

namespace shark {
/**
* \brief Implementation of the Pareto-Dominance relation under the assumption of all objectives to be minimized.
* \tparam Extractor returning the fitness vector of an object
*/
template<typename Extractor>
struct ParetoDominanceComparator {

	enum DominanceRelation{
		A_STRICTLY_DOMINATES_B = 3, ///< A strictly dominates B.
		A_WEAKLY_DOMINATES_B = 2, ///< A weakly dominates B.
		A_EQUALS_B = 1, ///< A equals B for every coordinate.
		TRADE_OFF = -1, ///< Both A and B are a valid trade-off.
		B_WEAKLY_DOMINATES_A = -2, ///< B weakly dominates B.
		B_STRICTLY_DOMINATES_A = -3, ///< B strictly dominates A.
	};
	

	/**
	* \brief Compares two individuals with respect to the Pareto-Dominance relation.
	* \tparam IndividualType The type of the individuals, needs to be a model of TypedIndividual.
	* \param [in] A Individual A.
	* \param [in] B Individual B.
	* \returns An integer with values according to the constanst defined within this class.
	*/
	template<typename IndividualType>
	int operator()( const IndividualType & A, const IndividualType & B ) {
		Extractor e;
		SIZE_CHECK(e( A ).size() == e( B ).size());
		
		unsigned numGreater = 0;
		unsigned numEqual = 0;
		unsigned numSmaller = 0;

		unsigned int noOfObj = e( A ).size();

		for (unsigned i = noOfObj; i--;) {
			if( e( A )[i] > e( B )[i] )
				numGreater++;
			else if( e( A )[i] < e( B )[i] )
				numSmaller++;
			else
				numEqual++;
		}

		if (numSmaller == noOfObj)
			return A_STRICTLY_DOMINATES_B; // A dominates B completely
		else if (numSmaller != 0 && numGreater == 0)
			return A_WEAKLY_DOMINATES_B; // A dominates B incompletely
		else if (numEqual == noOfObj)
			return A_EQUALS_B; // A equals B
		else if (numGreater == noOfObj)
			return B_STRICTLY_DOMINATES_A; // B dominates A completely
		else if (numGreater != 0 && numSmaller == 0)
			return B_WEAKLY_DOMINATES_A; // B dominates A incompletely

		return TRADE_OFF; // trade off
	}
};
}
#endif
