/*!
 * 
 *
 * \brief       BFGS
 * 
 * The Broyden, Fletcher, Goldfarb, Shannon (BFGS) algorithm is a
 * quasi-Newton method for unconstrained real-valued optimization.
 * 
 * 
 *
 * \author      O. Krause 
 * \date        2010
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
 #define SHARK_COMPILE_DLL
#include <shark/Algorithms/GradientDescent/AbstractLineSearchOptimizer.h>

using namespace shark;

AbstractLineSearchOptimizer::AbstractLineSearchOptimizer() {
	m_features |= REQUIRES_VALUE;
	m_features |= REQUIRES_FIRST_DERIVATIVE;
	m_linesearch.lineSearchType() = LineSearch::WolfeCubic;
}

void AbstractLineSearchOptimizer::init(ObjectiveFunctionType const& objectiveFunction, SearchPointType const& startingPoint) {
	checkFeatures(objectiveFunction);
	if(objectiveFunction.isConstrained()){
		//backtracking is the only alorithm that can handle constraints (e.g. initial bracketing phases are going to be nasty)
		m_linesearch.lineSearchType() = LineSearch::Backtracking;
		SHARK_RUNTIME_CHECK(objectiveFunction.isFeasible(startingPoint), "Initial point is not feasible");
	}

	m_linesearch.init(objectiveFunction);
	m_dimension = startingPoint.size();
	
	m_best.point = startingPoint;
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivative);
	

	// Get space for the different vectors we store.
	m_lastPoint.resize(m_dimension);
	m_searchDirection = -m_derivative;
	
	m_initialStepLength = 0.0;//1.0 as step length might be very wrong.
	for (size_t i = 0; i < m_derivative.size(); ++i)
		m_initialStepLength += std::abs(m_derivative(i));
	m_initialStepLength = std::min(1.0, 1.0 / m_initialStepLength);
	while(!objectiveFunction.isFeasible(m_best.point + m_initialStepLength * m_searchDirection)){
		m_initialStepLength /= 2.0;
	}
	
	initModel();
}

void AbstractLineSearchOptimizer::step(ObjectiveFunctionType const& objectiveFunction) {
	// Perform line search
	m_lastDerivative = m_derivative;
	m_lastPoint = m_best.point;
	m_lastValue = m_best.value;
	m_linesearch(m_best.point, m_best.value, m_searchDirection, m_derivative, m_initialStepLength);
	m_initialStepLength = 1.0;
	computeSearchDirection(objectiveFunction);
}

//from ISerializable
void AbstractLineSearchOptimizer::read(InArchive &archive) {
	archive>>m_linesearch;
	archive>>m_initialStepLength;
	archive>>m_dimension;
	archive>>m_best;
	archive>>m_derivative;
	archive>>m_searchDirection;
	archive>>m_lastDerivative;
	archive>>m_lastPoint;
	archive>>m_lastValue;
}

void AbstractLineSearchOptimizer::write(OutArchive &archive) const {
	archive<<m_linesearch;
	archive<<m_initialStepLength;
	archive<<m_dimension;
	archive<<m_best;
	archive<<m_derivative;
	archive<<m_searchDirection;
	archive<<m_lastDerivative;
	archive<<m_lastPoint;
	archive<<m_lastValue;
}
