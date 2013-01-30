//===========================================================================
/*!
* \brief Multi-objective optimization benchmark function LZ1.
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_LZ1_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_LZ1_H

#include <shark/Core/AbstractBoxConstraintsProvider.h>
#include <shark/ObjectiveFunctions/AbstractMultiObjectiveFunction.h>
#include <shark/Core/Traits/ObjectiveFunctionTraits.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

namespace shark {
	/*! \brief Multi-objective optimization benchmark function LZ1.
	*
	*  The function is described in
	*
	* H. Li and Q. Zhang. 
	* Multiobjective Optimization Problems with Complicated Pareto Sets, MOEA/D and NSGA-II, 
	* IEEE Trans on Evolutionary Computation, 2(12):284-302, April 2009. 
	*/
	struct LZ1 : 
		public AbstractMultiObjectiveFunction< VectorSpace<double> >,
		public TraitsBoxConstraintsProvider< VectorSpace<double>::PointType, LZ1 > {
			typedef AbstractMultiObjectiveFunction< VectorSpace<double> > super;
			typedef TraitsBoxConstraintsProvider< VectorSpace<double>::PointType, LZ1 > meta;

			typedef super::ResultType ResultType;
			typedef super::SearchPointType SearchPointType;	

			LZ1() : super( 2 ) {
				m_features |= CAN_PROPOSE_STARTING_POINT;
				m_features |= IS_CONSTRAINED_FEATURE;
				m_features |= CAN_PROVIDE_CLOSEST_FEASIBLE;
				m_name="LZ1";
			}

			void init() {
			}

			ResultType eval( const SearchPointType & x ) const {
				m_evaluationCounter++;

				ResultType value( 2, 0 );

				unsigned int counter1 = 0, counter2 = 0;
				for( unsigned int i = 1; i < x.size(); i++ ) {
					if( i % 2 == 0 ) {
						counter2++;
						value[1] += sqr( x(i) - ::pow( x(0), 0.5*(1.0+3*(i-1)/(x.size()-1) ) ) );
					} else {
						counter1++;
						value[0] += sqr( x(i) - ::pow( x(0), 0.5*(1.0+3*(i-1)/(x.size()-1) ) ) );
					}
				}

				value[0] *= 2./counter1;
				value[0] += x( 0 );

				value[1] *= 2./counter2;
				value[1] += 1 - ::sqrt( x( 0 ) );

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
	* \brief Specializes objective function traits for the function LZ1.
	*/
	template<> 
	struct ObjectiveFunctionTraits<LZ1> {

		static LZ1::SearchPointType lowerBounds( unsigned int n ) {
			LZ1::SearchPointType result( n, 0 );
			return( result );
		}

		static LZ1::SearchPointType upperBounds( unsigned int n ) {
			LZ1::SearchPointType result( n, 1 );
			return( result );
		}

	};

	ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( LZ1, shark::moo::RealValuedObjectiveFunctionFactory );
	//template<> struct ObjectiveFunctionTraits<LZ1> {
	//	static LZ1::SolutionSetType referenceSet( std::size_t maxSize,
	//		unsigned int numberOfVariables,
	//		unsigned int noObjectives ) {
	//		shark::IntervalIterator< tag::LinearTag > it( 0., 1., maxSize );
	//
	//		LZ1 lz1;
	//		lz1.numberOfVariables() = numberOfVariables;
	//
	//		LZ1::SolutionSetType solutionSet;
	//		while( it ) {
	//
	//			LZ1::SolutionType solution;
	//
	//			RealVector v( numberOfVariables );
	//			v( 0 ) = *it;
	//			for( unsigned int i = 1; i < numberOfVariables; i++ )
	//				v( i ) = ::pow( v(0), 0.5*(1.0+3*(i-1)/(numberOfVariables-1) ) );
	//
	//
	//			solution.searchPoint() = v;
	//			solution.objectiveFunctionValue() = lz1.eval( v );
	//			solutionSet.push_back( solution );
	//			++it;
	//		}
	//		return( solutionSet );
	//	}
	//
	//
	//	static LZ1::SearchPointType lowerBounds( unsigned int n ) {
	//		return( LZ1::SearchPointType( n, 0. ) );
	//	}
	//
	//	static LZ1::SearchPointType upperBounds( unsigned int n ) {
	//		return( LZ1::SearchPointType( n, 1. ) );
	//	}
	//
	//};
}
#endif
