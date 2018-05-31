/*!
 * 
 *
 * \brief       Implements a wrapper over an m_objective function which just rotates its inputs
 * 
 *
 * \author      O.Voss
 * \date        2010-2014
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_ROTATEDOBJECTIVEFUNCTION_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_ROTATEDOBJECTIVEFUNCTION_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/LinAlg/rotations.h>

namespace shark {namespace benchmarks{
///  \brief Rotates an objective function using a randomly initialized rotation.
///
/// Most benchmark functions are axis aligned because it is assumed that the algorithm
/// is rotation invariant. However this does not mean that all its aspects are the same.
/// Especially linear algebra routines might take longer when the problem is not
/// axis aligned. This function creates a random rotation function and 
/// applies it to the given input points to make it no longer axis aligned.
///  \ingroup benchmarks
struct RotatedObjectiveFunction : public SingleObjectiveFunction {
	RotatedObjectiveFunction(SingleObjectiveFunction* objective)
	:m_objective(objective){
		if(m_objective->canProposeStartingPoint())
			m_features |= CAN_PROPOSE_STARTING_POINT;
		if(m_objective->hasFirstDerivative())
			m_features |= HAS_FIRST_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RotatedObjectiveFunction<"+m_objective->name()+">"; }
	
	std::size_t numberOfVariables()const{
		return m_objective->numberOfVariables();
	}
	
	void init(){
		m_rotation = blas::randomRotationMatrix(*mep_rng, numberOfVariables());
		m_objective->setRng(mep_rng);
		m_objective->init();
	}
	
	bool hasScalableDimensionality()const{
		return m_objective->hasScalableDimensionality();
	}
	
	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_objective->setNumberOfVariables(numberOfVariables);
	}

	SearchPointType proposeStartingPoint() const {
		RealVector y = m_objective->proposeStartingPoint();
		
		return prod(trans(m_rotation),y);
	}

	double eval( SearchPointType const& p ) const {
		m_evaluationCounter++;
		RealVector x = prod(m_rotation,p);
		return m_objective->eval(x);
	}
	
	ResultType evalDerivative( SearchPointType const& p, FirstOrderDerivative& derivative )const {
		RealVector x = prod(m_rotation,p);
		double value = m_objective->evalDerivative(x,derivative);
		derivative = prod(trans(m_rotation),derivative);
		return value;
	}
private:
	SingleObjectiveFunction* m_objective;
	RealMatrix m_rotation;
};

}}

#endif
