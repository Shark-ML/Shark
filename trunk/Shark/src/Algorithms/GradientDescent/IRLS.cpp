/*!
 *  \file IRLS.cpp
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
#include <shark/Algorithms/GradientDescent/IRLS.h>
#include <shark/LinAlg/solveSystem.h>

using namespace shark;

IRLS::IRLS()
{
	m_features |= REQUIRES_VALUE;
	m_features |= REQUIRES_FIRST_DERIVATIVE;
	m_features |= REQUIRES_SECOND_DERIVATIVE;
	m_linesearch.lineSearchType() = LineSearch::Linmin;
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