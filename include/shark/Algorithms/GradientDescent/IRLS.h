/*!
 *  \file IRLS.h
 *
 *  \brief simple Newton step method
 *
 * 
 *
 *  \author O. Krause
 *  \date 2010
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef ALGORITHMS_TRAINER_IRLS_H
#define ALGORITHMS_TRAINER_IRLS_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Algorithms/GradientDescent/LineSearch.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

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
class IRLS : public AbstractSingleObjectiveOptimizer<VectorSpace<double> >
{
public:
	IRLS();
	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint);
	using AbstractSingleObjectiveOptimizer<VectorSpace<double> >::init;
	
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
