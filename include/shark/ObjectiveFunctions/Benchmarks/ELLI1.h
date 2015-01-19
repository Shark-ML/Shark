//===========================================================================
/*!
 * 
 *
 * \brief       Multi-objective optimization benchmark function ELLI 1.
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
//===========================================================================
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_ELLI1_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_ELLI1_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/Rng/GlobalRng.h>

#include <shark/LinAlg/rotations.h>

namespace shark {
/*! \brief Multi-objective optimization benchmark function ELLI1.
*
*  The function is described in
*
*  Christian Igel, Nikolaus Hansen, and Stefan Roth. 
*  Covariance Matrix Adaptation for Multi-objective Optimization. 
*  Evolutionary Computation 15(1), pp. 1-28, 2007 
*/
struct ELLI1 : public MultiObjectiveFunction{
	
	ELLI1(std::size_t numVariables = 0) : m_a( 1E6 ){
		m_features |= CAN_PROPOSE_STARTING_POINT;
		setNumberOfVariables(numVariables);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ELLI1"; }

	std::size_t numberOfObjectives()const{
		return 2;
	}
	
	std::size_t numberOfVariables()const{
		return m_coefficients.size();
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}

	void setNumberOfVariables( std::size_t numVariables ){
		m_coefficients.resize(numVariables);
		for(std::size_t i = 0; i != numVariables; ++i){
			m_coefficients(i) = std::pow(m_a, 2.0 * (i / (numVariables - 1.0)));
		}
	}

	void init() {
		m_rotationMatrix = blas::randomRotationMatrix( numberOfVariables() );
	}

	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		ResultType value( 2 );

		//point_type point = m_rotationMatrix * x;
		SearchPointType y = prod( m_rotationMatrix, x );

		double sum1 = 0.0;
		double sum2 = 0.0;
		for (unsigned i = 0; i < numberOfVariables(); i++) {
			sum1 += m_coefficients(i) * sqr( y(i) );
			sum2 += m_coefficients(i) * sqr( y(i) - 2.0 );
		}

		value[0] = sum1 / ( sqr(m_a) * numberOfVariables() );
		value[1] = sum2 / ( sqr(m_a) * numberOfVariables() );

		return value;
	}

	void proposeStartingPoint( SearchPointType & x ) const {
		x.resize( numberOfVariables() );
		for( unsigned int i = 0; i < numberOfVariables(); i++ )
			x( i ) = Rng::uni( -10., 10. );
	}

private:
	double m_a;
	RealMatrix m_rotationMatrix;
	RealVector m_coefficients;
};

}
#endif
