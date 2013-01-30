//===========================================================================
/*!
* \file ELLI1.h
*
* \brief Multi-objective optimization benchmark function ELLI 1.
*
*  The function is described in
*
*  Christian Igel, Nikolaus Hansen, and Stefan Roth. 
*  Covariance Matrix Adaptation for Multi-objective Optimization. 
*  Evolutionary Computation 15(1), pp. 1-28, 2007
*
* <BR><HR>
* This file is part of Shark. This library is free software;
* you can redistribute it and/or modify it under the terms of the
* GNU General Public License as published by the Free Software
* Foundation; either version 3, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this library; if not, see <http://www.gnu.org/licenses/>.
*/
//===========================================================================
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_ELLI1_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_ELLI1_H

#include <shark/Core/AbstractBoxConstraintsProvider.h>
#include <shark/ObjectiveFunctions/AbstractMultiObjectiveFunction.h>
#include <shark/Core/Traits/ObjectiveFunctionTraits.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

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
	struct ELLI1 : public AbstractMultiObjectiveFunction< VectorSpace<double> > {

		typedef AbstractMultiObjectiveFunction< VectorSpace<double> > super;

		typedef super::ResultType ResultType;
		typedef super::SearchPointType SearchPointType;

		

		ELLI1() : m_a( 1E6 ) {
			m_features |= CAN_PROPOSE_STARTING_POINT;
			m_name = "ELLI1";
		}

		void init() {
			m_rotationMatrix = randomRotationMatrix( m_numberOfVariables );
		}

		ResultType eval( const SearchPointType & x ) const {
			m_evaluationCounter++;

			ResultType value( 2 );

			//point_type point = m_rotationMatrix * x;
			SearchPointType y = blas::prod( m_rotationMatrix, x );

			double sum1 = 0.0, sum2 = 0.0;

			for (unsigned i = 0; i < numberOfVariables(); i++) {
				sum1 += std::pow(m_a, 2.0 * (i / (numberOfVariables() - 1.0))) * y( i ) * y( i );
				sum2 += std::pow(m_a, 2 * (i / (numberOfVariables() - 1.0))) * (y( i ) - 2.0) * (y( i ) - 2.0);
			}

			value[0] = 2 + sum1 / ( m_a * m_a * numberOfVariables() );
			value[1] = 2 + sum2 / ( m_a * m_a * numberOfVariables() );

			return( value );
		}

		void proposeStartingPoint( SearchPointType & x ) const {
			x.resize( m_numberOfVariables );
			for( unsigned int i = 0; i < m_numberOfVariables; i++ )
				x( i ) = Rng::gauss( -10., 10. );
		}

	private:
		double m_a;
		RealMatrix m_rotationMatrix;
	};

	ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( ELLI1, shark::moo::RealValuedObjectiveFunctionFactory );
}
#endif
