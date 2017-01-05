/*!
 * 
 *
 * \brief       Implementation of the Pareto-Dominance relation.
 * 
 *
 * \author      T. Glasmachers (based on old version by T. Voß)
 * \date        2011-2016
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_DOMINATION_PARETODOMINANCE_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_DOMINATION_PARETODOMINANCE_H


#include <shark/LinAlg/Base.h>


namespace shark {


/// \brief Result of comparing two objective vectors w.r.t. Pareto dominance.
enum DominanceRelation
{
	INCOMPARABLE = 0,       // LHS and RHS are incomparable
	LHS_DOMINATES_RHS = 1,  // LHS strictly dominates RHS
	RHS_DOMINATES_LHS = 2,  // RHS strictly dominates LHS
	EQUIVALENT = 3,         // LHS and RHS are equally good
};

/// \brief Pareto dominance relation for two objective vectors.
template<class VectorTypeA, class VectorTypeB>
inline DominanceRelation dominance(VectorTypeA const& lhs, VectorTypeB const& rhs)
{
	SHARK_ASSERT(lhs.size() == rhs.size());
	std::size_t l = 0, r = 0;
	for (std::size_t i=0; i<lhs.size(); i++)
	{
		if (lhs(i) < rhs(i)) l++;
		else if (lhs(i) > rhs(i)) r++;
	}

	if (l > 0)
	{
		if (r > 0) return INCOMPARABLE;
		else return LHS_DOMINATES_RHS;
	}
	else
	{
		if (r > 0) return RHS_DOMINATES_LHS;
		else return EQUIVALENT;
	}
}


};  // namespace shark
#endif
