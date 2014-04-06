/*!
 * 
 *
 * \brief       simple Newton step method
 * 
 * 
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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

#ifndef ALGORITHMS_TRAINER_IRLS_H
#define ALGORITHMS_TRAINER_IRLS_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Algorithms/GradientDescent/LineSearch.h>

namespace shark {

///\brief Iterated Reweightes Least Squares iteratively calculates a newton step and then performs a line search in that direction
///
///IRLS tries to iteratively approximate the function as a quadratic function. Than it simply 
///performs a newton step to calculate the optimal solution to the quadratic problem. Since the 
///function itself is not quadratic most of the time, IRLS then performs a line search in that 
///direction instead of just taking the newton solution as optimal point. This prevents divergence.
///Be warned that calculating the inverse hessian takes O(n^2) 
/// memory tand O(n^3) runtime. So for large scale problems
///use CG or BFGS instead.
///TODO: implement backtracking linesearch for this 
class IRLS : public AbstractSingleObjectiveOptimizer<RealVector >
{
public:
	IRLS();
	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint);
	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	
	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "IRLS"; }

	void hessianIsPositiveDefinite(bool isPositive){
		m_isPositive= true;
	}

	void step(const ObjectiveFunctionType& objectiveFunction);
	const LineSearch& lineSearch()const
	{
		return m_linesearch;
	}
	LineSearch& lineSearch()
	{
		return m_linesearch;
	}
protected:
	bool m_isPositive;
	ObjectiveFunctionType::SecondOrderDerivative m_derivatives;
	LineSearch m_linesearch;
};
}
#endif
