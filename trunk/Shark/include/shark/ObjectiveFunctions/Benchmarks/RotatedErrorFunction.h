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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_ROTATEDOBJECTIVEFUNCTION_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_ROTATEDOBJECTIVEFUNCTION_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/LinAlg/rotations.h>

namespace shark {
///  \brief Rotates an objective function using a randomly initialized rotation.
///
/// Most benchmark functions are axis aligned because it is assumed that the algorithm
/// is rotation invariant. However this does not mean that all its aspects are the same.
/// Especially linear algebra routines might take longer when the problem is not
/// axis aligned. This function creates a random rotation function and 
/// applies it to the given input points to make it no longer axis aligned.
struct RotatedObjectiveFunction : public SingleObjectiveFunction {
	RotatedObjectiveFunction(SingleObjectiveFunction* objective)
	:m_objective(objective){
		if(m_objective->canProposeStartingPoint())
			m_features |= CAN_PROPOSE_STARTING_POINT;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RotatedObjectiveFunction<"+m_objective->name()+">"; }
	
	std::size_t numberOfVariables()const{
		return m_objective->numberOfVariables();
	}
	
	void init(){
		m_rotation = blas::randomRotationMatrix(numberOfVariables());
		m_objective->init();
	}
	
	bool hasScalableDimensionality()const{
		return m_objective->hasScalableDimensionality();
	}
	
	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_objective->setNumberOfVariables(numberOfVariables);
	}

	void configure( const PropertyTree & node ) {
		m_objective->configure(node);
	}

	void proposeStartingPoint( SearchPointType & x ) const {
		RealVector y(numberOfVariables());
		m_objective->proposeStartingPoint(y);
		
		x.resize( numberOfVariables() );
		axpy_prod(trans(m_rotation),y,x,true);
	}

	double eval( const SearchPointType & p ) const {
		m_evaluationCounter++;
		RealVector x(numberOfVariables());
		axpy_prod(m_rotation,p,x);
		return m_objective->eval(x);
	}
private:
	SingleObjectiveFunction* m_objective;
	RealMatrix m_rotation;
};

}

#endif
