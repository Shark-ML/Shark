//===========================================================================
/*!
 * 
 *
 * \brief       Multi-objective optimization benchmark function CIGTAB 2.
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_CIGTAB2_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_CIGTAB2_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

#include <shark/LinAlg/rotations.h>

namespace shark {
/*! \brief Multi-objective optimization benchmark function CIGTAB 2.
*
*  The function is described in
*
*  Christian Igel, Nikolaus Hansen, and Stefan Roth. 
*  Covariance Matrix Adaptation for Multi-objective Optimization. 
*  Evolutionary Computation 15(1), pp. 1-28, 2007 
*/
struct CIGTAB2 : public MultiObjectiveFunction {

	CIGTAB2(std::size_t numberOfVariables = 5) : m_a( 1E-6 ) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_numberOfVariables = numberOfVariables;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CIGTAB2"; }

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

	void configure( const PropertyTree & node ) {
		m_a = node.get( "a", 1E6);
	}

	void init() {
		m_rotationMatrixY = blas::randomRotationMatrix(m_numberOfVariables);
		m_rotationMatrixZ = blas::randomRotationMatrix(m_numberOfVariables);
	}

	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		ResultType value( 2 );

		SearchPointType y = blas::prod( m_rotationMatrixY, x );
		SearchPointType z = blas::prod( m_rotationMatrixZ, x );
		double result_1 = y(0) * y(0) + m_a * m_a * y(numberOfVariables()-1) * y(numberOfVariables()-1);
		double result_2 = z(0) * z(0) + m_a * m_a * z(numberOfVariables()-1) * z(numberOfVariables()-1);

		for (unsigned i = 1; i < numberOfVariables() - 1; i++) {
			result_1 += m_a * y( i ) * y( i );
			result_2 += m_a * (z( i ) - 2) * (z( i ) - 2);
		}

		value[0] = result_1 / (m_a * m_a * numberOfVariables());
		value[1] = result_2 / (m_a * m_a * numberOfVariables());

		return value;
	}

	void proposeStartingPoint( SearchPointType & x ) const {
		x.resize( m_numberOfVariables );
		for( unsigned int i = 0; i < m_numberOfVariables; i++ )
			x( i ) = Rng::uni( -10., 10. );
	}
private:
	double m_a;
	std::size_t m_numberOfVariables;
	RealMatrix m_rotationMatrixY;
	RealMatrix m_rotationMatrixZ;
};
}
#endif
