//===========================================================================
/*!
* \file LZ6.h
*
* \brief Multi-objective optimization benchmark function LZ6.
*
*  The function is described in
*
* H. Li and Q. Zhang. 
* Multiobjective Optimization Problems with Complicated Pareto Sets, MOEA/D and NSGA-II, 
* IEEE Trans on Evolutionary Computation, 2(12):284-302, April 2009. 
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_LZ6_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_LZ6_H

#include <shark/Core/AbstractBoxConstraintsProvider.h>
#include <shark/ObjectiveFunctions/AbstractMultiObjectiveFunction.h>
#include <shark/Core/Traits/ObjectiveFunctionTraits.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

namespace shark {
	/*! \brief Multi-objective optimization benchmark function LZ6.
	*
	*  The function is described in
	*
	*  H. Li and Q. Zhang. 
	*  Multiobjective Optimization Problems with Complicated Pareto Sets, MOEA/D and NSGA-II, 
	*  IEEE Trans on Evolutionary Computation, 2(12):284-302, April 2009. 
	*/
	struct LZ6 : 
		public AbstractMultiObjectiveFunction< VectorSpace<double> >,
		public TraitsBoxConstraintsProvider< VectorSpace<double>::PointType, LZ6 > {
			typedef AbstractMultiObjectiveFunction< VectorSpace<double> > super;
			typedef TraitsBoxConstraintsProvider< VectorSpace<double>::PointType, LZ6 > meta;

			typedef super::ResultType ResultType;
			typedef super::SearchPointType SearchPointType;

			

			LZ6() : super( 3 ) {
				m_features |= CAN_PROPOSE_STARTING_POINT;
				m_features |= IS_CONSTRAINED_FEATURE;
				m_features |= CAN_PROVIDE_CLOSEST_FEASIBLE;
				m_name="LZ6";
			}

			void init() {
			}

			ResultType eval( const SearchPointType & x ) const {
				m_evaluationCounter++;

				ResultType value( 3, 0 );

				unsigned int counter1 = 0, counter2 = 0, counter3 = 0;
				for( unsigned int i = 3; i <= x.size(); i++ ) {
					if( (i-1) % 3 == 0 ) { //J1
						counter1++;
						value[0] += sqr(x(i-1)-2*x( 1 )*::sin( 2 * M_PI * x( 0 ) + i*M_PI/x.size() ) );
					} else if( (i-2) % 3 == 0 ) { //J2
						counter2++;
						value[1] += sqr(x(i-1)-2*x( 1 )*::sin( 2 * M_PI * x( 0 ) + i*M_PI/x.size() ) );
					} else if( i % 3 == 0 ) {
						counter3++;
						value[2] += sqr(x(i-1)-2*x( 1 )*::sin( 2 * M_PI * x( 0 ) + i*M_PI/x.size() ) );
					}
				}

				value[0] *= counter1 > 0 ? 2./counter1 : 1;
				value[1] *= counter2 > 0 ? 2./counter2 : 1;
				value[2] *= counter3 > 0 ? 2./counter3 : 1;

				value[0] += ::cos( 0.5*M_PI * x( 0 ) ) * ::cos( 0.5*M_PI * x( 1 ) );
				value[1] += ::cos( 0.5*M_PI * x( 0 ) ) * ::sin( 0.5*M_PI * x( 1 ) );
				value[2] += ::sin( 0.5*M_PI * x( 0 ) );

				return( value );
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
	};

	/**
	* \brief Specializes objective function traits for the function LZ6.
	*/
	template<> 
	struct ObjectiveFunctionTraits<LZ6> {

		static LZ6::SearchPointType lowerBounds( unsigned int n ) {
			LZ6::SearchPointType result( n, -2 );
			result( 0 ) = 0;
			result( 1 ) = 0;
			return( result );
		}

		static LZ6::SearchPointType upperBounds( unsigned int n ) {
			LZ6::SearchPointType result( n, 2 );
			result( 0 ) = 1;
			result( 1 ) = 1;
			return( result );
		}

	};

	ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( LZ6, shark::moo::RealValuedObjectiveFunctionFactory );

	//template<> struct ObjectiveFunctionTraits<LZ6> {
	//	static LZ6::SolutionSetType referenceSet( std::size_t maxSize,
	//		unsigned int numberOfVariables,
	//		unsigned int noObjectives ) {
	//			shark::IntervalIterator< tag::LinearTag > it( 0., 1., maxSize );
	//
	//			LZ6 lz6;
	//			lz6.numberOfVariables() = numberOfVariables;
	//
	//			LZ6::SolutionSetType solutionSet;
	//			while( it ) {
	//
	//				shark::IntervalIterator< tag::LinearTag > itt( 0., 1., maxSize );
	//
	//				while( itt ) {
	//					LZ6::SolutionType solution;
	//
	//					RealVector v( numberOfVariables );
	//					v( 0 ) = *it;
	//					v( 1 ) = *itt;
	//					for( unsigned int i = 3; i <= numberOfVariables; i++ ) {
	//						v( i-1 ) = 2 * v( 1 ) * ::sin( 2 * M_PI * v( 0 ) + i*M_PI/numberOfVariables);
	//					}
	//
	//					solution.searchPoint() = v;
	//					solution.objectiveFunctionValue() = lz6.eval( v );
	//					solutionSet.push_back( solution );
	//					++itt;
	//				}
	//				++it;
	//			}
	//			return( solutionSet );
	//	}
	//
	//
	//	static LZ6::SearchPointType lowerBounds( unsigned int n ) {
	//		LZ6::SearchPointType sp( n, -1. );
	//		sp( 0 ) = 0;
	//		sp( 1 ) = 0;
	//
	//		return( sp );
	//	}
	//
	//	static LZ6::SearchPointType upperBounds( unsigned int n ) {
	//		return( LZ6::SearchPointType( n, 1. ) );
	//	}
	//
	//};
}
#endif
