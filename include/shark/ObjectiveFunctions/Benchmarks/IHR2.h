//===========================================================================
/*!
* \brief Multi-objective optimization benchmark function IHR 2.
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_IHR2_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_IHR2_H

#include <shark/Core/AbstractBoxConstraintsProvider.h>
#include <shark/ObjectiveFunctions/AbstractMultiObjectiveFunction.h>
#include <shark/Core/Traits/ObjectiveFunctionTraits.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

#include <shark/LinAlg/rotations.h>

#include <vector>

namespace shark{
	/*! \brief Multi-objective optimization benchmark function IHR 2.
	*
	*  The function is described in
	*
	*  Christian Igel, Nikolaus Hansen, and Stefan Roth. 
	*  Covariance Matrix Adaptation for Multi-objective Optimization. 
	*  Evolutionary Computation 15(1), pp. 1-28, 2007 
	*/
	struct IHR2 : public AbstractMultiObjectiveFunction< VectorSpace<double> >,
		public TraitsBoxConstraintsProvider< VectorSpace<double>::PointType, IHR2 > {
			typedef AbstractMultiObjectiveFunction< VectorSpace<double> > super;
			typedef TraitsBoxConstraintsProvider< VectorSpace<double>::PointType, IHR2 > meta;

			typedef super::ResultType ResultType;
			typedef super::SearchPointType SearchPointType;

			

			IHR2() : super( 2 ), m_a( 1000 ) {
				m_name = "IHR2";
				m_features |= CAN_PROPOSE_STARTING_POINT;
				m_features |= IS_CONSTRAINED_FEATURE;
				m_features |= CAN_PROVIDE_CLOSEST_FEASIBLE;
			}

			void init() {
				m_rotationMatrix = randomRotationMatrix(m_numberOfVariables);
			}

			ResultType eval( const SearchPointType & x )const {
				m_evaluationCounter++;

				ResultType value( 2 );

				SearchPointType y = prod(m_rotationMatrix,x);

				value[0] = ::fabs( y( 0 ) );

				double g = 0;
				double ymax = ::fabs( m_rotationMatrix(0, 0) );

				for( unsigned int i = 1; i < numberOfVariables(); i++ )
					ymax = std::max( ::fabs( m_rotationMatrix(0, i) ), ymax );
				ymax = 1. / ymax;

				for (unsigned i = 1; i < numberOfVariables(); i++)
					g += hg( y( i ) );
				g = 9 * g / (numberOfVariables() - 1.) + 1.;


				value[1] = g * hf(1. - sqr(y( 0 ) / g), y( 0 ), ymax);

				return value;
			}

			void proposeStartingPoint( SearchPointType & x ) const {
				meta::proposeStartingPoint( x, m_numberOfVariables );
			}

			bool isFeasible( const SearchPointType & v ) const {
				return( meta::isFeasible( v ) );
			}

			void closestFeasible( SearchPointType & v ) const {
				meta::closestFeasible( v );
			}

			double h( double x, double n )const {
				return 1 / ( 1 + ::exp( -x / ::sqrt( n ) ) );
			}

			double hf(double x, double y0, double ymax)const {
				if( ::fabs(y0) <= ymax )
					return x;
				return ::fabs( y0 ) + 1.;
			}

			double hg(double x)const {
				return (x*x) / ( ::fabs(x) + 0.1 );
			}
	private:
		double m_a;
		RealMatrix m_rotationMatrix;
	};

	/**
	* \brief Specializes objective function traits for the function IHR2.
	*/
	template<> 
	struct ObjectiveFunctionTraits<IHR2> {

		static IHR2::SearchPointType lowerBounds( unsigned int n ) {
			IHR2::SearchPointType result( n, -1 );
			return( result );
		}

		static IHR2::SearchPointType upperBounds( unsigned int n ) {
			IHR2::SearchPointType result( n, 1 );
			return( result );
		}

	};

	ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( IHR2, shark::moo::RealValuedObjectiveFunctionFactory );
}
#endif
