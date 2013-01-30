//===========================================================================
/*!
* \brief Multi-objective optimization benchmark function CIGTAB 1.
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_CIGTAB1_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_CIGTAB1_H

#include <shark/Core/AbstractBoxConstraintsProvider.h>
#include <shark/ObjectiveFunctions/AbstractMultiObjectiveFunction.h>
#include <shark/Core/Traits/ObjectiveFunctionTraits.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

#include <shark/LinAlg/rotations.h>

#include <vector>

namespace shark {
	/*! \brief Multi-objective optimization benchmark function CIGTAB 1.
	*
	*  The function is described in
	*
	*  Christian Igel, Nikolaus Hansen, and Stefan Roth. 
	*  Covariance Matrix Adaptation for Multi-objective Optimization. 
	*  Evolutionary Computation 15(1), pp. 1-28, 2007 
	*/
	struct CIGTAB1 : public AbstractMultiObjectiveFunction< VectorSpace<double> > {

		typedef AbstractMultiObjectiveFunction< VectorSpace<double> > super;

		typedef super::ResultType ResultType;
		typedef super::SearchPointType SearchPointType;

		

		CIGTAB1() : super( 2 ), m_a( 1E6 ) {
			m_name = "CIGTAB1";
			m_features |= CAN_PROPOSE_STARTING_POINT;
		}

		void configure( const PropertyTree & node ) {
			m_a = node.get( "a", 1E6 );
		}

		void init() {
			m_rotationMatrix = randomRotationMatrix(m_numberOfVariables);
		}

		ResultType eval( const SearchPointType & x ) const {
			m_evaluationCounter++;

			ResultType value( 2 );

			//point_type point = m_rotationMatrix % x;
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

		void proposeStartingPoint( SearchPointType & x ) const {
			x.resize( m_numberOfVariables );
			for( unsigned int i = 0; i < m_numberOfVariables; i++ )
				x( i ) = Rng::gauss( -10., 10. );
		}
	private:
		double m_a;
		RealMatrix m_rotationMatrix;
	};

	ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( CIGTAB1, shark::moo::RealValuedObjectiveFunctionFactory );
}
#endif
