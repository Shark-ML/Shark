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
#include <shark/Algorithms/GradientDescent/IRLS.h>
#include <shark/LinAlg/solveSystem.h>

using namespace shark;

IRLS::IRLS()
{
	m_features |= REQUIRES_VALUE;
	m_features |= REQUIRES_FIRST_DERIVATIVE;
	m_features |= REQUIRES_SECOND_DERIVATIVE;
	m_isPositive = false;
}
void IRLS::init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint) {
	checkFeatures(objectiveFunction);
	m_best.point = startingPoint;
	m_linesearch.init(objectiveFunction);
	
	///valuate initial point
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivatives);
}


void IRLS::step(const ObjectiveFunctionType& objectiveFunction) {
	//calculate search direction
	RealVector searchDirection;
	if(m_isPositive)
		blas::solveSymmSystem<blas::SolveAXB>(m_derivatives.m_hessian,searchDirection, -m_derivatives.m_gradient);
	else
		blas::solveSystem(m_derivatives.m_hessian,searchDirection, -m_derivatives.m_gradient);
	
	//perform a line search in the newton direction
	m_linesearch(m_best.point,m_best.value,searchDirection,m_derivatives.m_gradient);
	
	//evaluate new point
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivatives);
}