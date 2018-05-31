//===========================================================================
/*!
 * 
 *
 * \brief       Multi-objective optimization benchmark function CIGTAB 1.
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
//===========================================================================
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_CIGTAB1_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_CIGTAB1_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/LinAlg/rotations.h>


namespace shark {namespace benchmarks{
/*! \brief Multi-objective optimization benchmark function CIGTAB 1.
*
*  The function is described in
*
*  Christian Igel, Nikolaus Hansen, and Stefan Roth. 
*  Covariance Matrix Adaptation for Multi-objective Optimization. 
*  Evolutionary Computation 15(1), pp. 1-28, 2007
* \ingroup benchmarks
*/
struct CIGTAB1 : public MultiObjectiveFunction {

	CIGTAB1(std::size_t numberOfVariables = 5) : m_a( 1E6 ) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_numberOfVariables = numberOfVariables;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CIGTAB1"; }

	std::size_t numberOfObjectives()const{
		return 2;
	}
	
	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}
	
	/// \brief Adjusts the number of variables if the function is scalable.
	/// \param [in] numberOfVariables The new dimension.
	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
	}

	void init() {
		m_rotationMatrix = blas::randomRotationMatrix(*mep_rng, m_numberOfVariables);
	}

	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		ResultType value(2);

		ResultType y = prod( m_rotationMatrix, x );
		double result = sqr( y(0) ) + sqr( m_a ) * sqr( y( numberOfVariables() - 1 ) );

		for (unsigned i = 1; i < numberOfVariables() - 1; i++) {
			result += m_a * sqr( y( i ) );
		}

		value[0] = result / ( sqr(m_a) * numberOfVariables() );

		result = sqr(y( 0 ) - 2) + sqr(m_a) * sqr(y(numberOfVariables()-1) - 2);

		for (unsigned i = 1; i < numberOfVariables() - 1; i++) {
			result += m_a * sqr(y( i ) - 2);
		}

		value[1] = result / ( sqr(m_a) * numberOfVariables() );

		return value;
	}

	SearchPointType proposeStartingPoint() const {
		RealVector x(m_numberOfVariables);

		for (std::size_t i = 0; i < x.size(); i++) {
			x(i) = random::uni(*mep_rng, -10.0, 10.0);
		}
		return x;
	}
private:
	double m_a;
	RealMatrix m_rotationMatrix;
	std::size_t m_numberOfVariables;
};
}}
#endif
