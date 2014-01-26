//===========================================================================
/*!
 * 
 *
 * \brief       Multi-objective optimization benchmark function ELLI 2.
 * 
 * The function is described in
 * 
 * Christian Igel, Nikolaus Hansen, and Stefan Roth. 
 * Covariance Matrix Adaptation for Multi-objective Optimization. 
 * Evolutionary Computation 15(1), pp. 1-28, 2007
 * 
 * 
 *
 * \author      -
 * \date        -
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
//===========================================================================
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_ELLI2_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_ELLI2_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/Rng/GlobalRng.h>

#include <shark/LinAlg/rotations.h>

namespace shark {
/*! \brief Multi-objective optimization benchmark function ELLI2.
*
*  The function is described in
*
*  Christian Igel, Nikolaus Hansen, and Stefan Roth. 
*  Covariance Matrix Adaptation for Multi-objective Optimization. 
*  Evolutionary Computation 15(1), pp. 1-28, 2007 
*/
struct ELLI2 : public MultiObjectiveFunction{

	ELLI2() : m_a( 1E6 ) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ELLI2"; }

	std::size_t numberOfObjectives()const{
		return 2;
	}
	
	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}

	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
	}

	void init() {
		m_rotationMatrixX = blas::randomRotationMatrix( m_numberOfVariables );
		m_rotationMatrixY = blas::randomRotationMatrix( m_numberOfVariables );
	}

	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		ResultType value( 2 );

		double sum_1 = 0, sum_2 = 0;

		SearchPointType y = blas::prod( m_rotationMatrixX, x );
		SearchPointType z = blas::prod( m_rotationMatrixY, x );

		for (unsigned i = 0; i < numberOfVariables(); i++) {
			sum_1 += std::pow(m_a, 2.0 * (i / (numberOfVariables() - 1.0))) * y(i) * y(i);
			sum_2 += std::pow(m_a, 2.0 * (i / (numberOfVariables() - 1.0))) * (z(i) - 2.0) * (z(i) - 2.0);
		}

		value[0] = sum_1 / (m_a * m_a * numberOfVariables());
		value[1] = sum_2 / (m_a * m_a * numberOfVariables());

		return value;
	}

	void proposeStartingPoint( SearchPointType & x ) const {
		x.resize( m_numberOfVariables );
		for( unsigned int i = 0; i < m_numberOfVariables; i++ )
			x( i ) = Rng::gauss( -10., 10. );
	}
private:
	double m_a;
	RealMatrix m_rotationMatrixX;
	RealMatrix m_rotationMatrixY;
	std::size_t m_numberOfVariables;
};

ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( ELLI2, shark::moo::RealValuedObjectiveFunctionFactory );
}
#endif
