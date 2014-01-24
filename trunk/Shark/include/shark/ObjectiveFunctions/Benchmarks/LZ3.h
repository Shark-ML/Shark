//===========================================================================
/*!
 * 
 * \file        LZ3.h
 *
 * \brief       Multi-objective optimization benchmark function LZ3.
 * 
 * The function is described in
 * 
 * H. Li and Q. Zhang. 
 * Multiobjective Optimization Problems with Complicated Pareto Sets, MOEA/D and NSGA-II, 
 * IEEE Trans on Evolutionary Computation, 2(12):284-302, April 2009. 
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_LZ3_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_LZ3_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

namespace shark {
/*! \brief Multi-objective optimization benchmark function LZ3.
*
*  The function is described in
*
*  H. Li and Q. Zhang. 
*  Multiobjective Optimization Problems with Complicated Pareto Sets, MOEA/D and NSGA-II, 
*  IEEE Trans on Evolutionary Computation, 2(12):284-302, April 2009. 
*/
struct LZ3 : public MultiObjectiveFunction
{
	LZ3(std::size_t numVariables = 0) :m_handler(SearchPointType(numVariables,-1),SearchPointType(numVariables,1) ){
		announceConstraintHandler(&m_handler);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LZ3"; }

	std::size_t numberOfObjectives()const{
		return 2;
	}
	
	std::size_t numberOfVariables()const{
		return m_handler.dimensions();
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}

	/// \brief Adjusts the number of variables if the function is scalable.
	/// \param [in] numberOfVariables The new dimension.
	void setNumberOfVariables( std::size_t numberOfVariables ){
		SearchPointType lb(numberOfVariables,-1);
		SearchPointType ub(numberOfVariables, 1);
		lb(0) = 0;
		ub(0) = 1;
		m_handler.setBounds(lb, ub);
	}

	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		ResultType value( 2, 0 );

		unsigned int counter1 = 0, counter2 = 0;
		for( unsigned int i = 1; i < x.size(); i++ ) {
			if( i % 2 == 0 ) {
				counter2++;
				value[1] += sqr( x(i) - 0.8*x( 0 )*::cos( 6 * M_PI * x( 0 ) + (i-1)*M_PI/(x.size()-1) ) );
			} else {
				counter1++;
				value[0] += sqr( x(i) - 0.8*x( 0 )*::sin( 6 * M_PI * x( 0 ) + (i-1)*M_PI/(x.size()-1) ) );
			}
		}

		value[0] *= 2./counter1;
		value[0] += x( 0 );

		value[1] *= 2./counter2;
		value[1] += 1 - ::sqrt( x( 0 ) );

		return( value );
	}
private:
	BoxConstraintHandler<SearchPointType> m_handler;
};

ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( LZ3, shark::moo::RealValuedObjectiveFunctionFactory );
	//template<> struct ObjectiveFunctionTraits<LZ3> {
	//	static LZ3::SolutionSetType referenceSet( std::size_t maxSize,
	//		unsigned int numberOfVariables,
	//		unsigned int numberOfObjectives ) {
	//		shark::IntervalIterator< tag::LinearTag > it( 0., 1., maxSize );
	//
	//		LZ3 lz3;
	//		lz3.numberOfVariables() = numberOfVariables;
	//
	//		LZ3::SolutionSetType solutionSet;
	//		while( it ) {
	//
	//			LZ3::SolutionType solution;
	//
	//			RealVector v( numberOfVariables );
	//			v( 0 ) = *it;
	//			for( unsigned int i = 1; i < numberOfVariables; i++ ) {
	//				if( i % 2 == 0 )
	//					v( i ) = 0.8*v( 0 )*::cos( 6 * M_PI * v( 0 ) + (i-1)*M_PI/(v.size()-1) );
	//				if( i % 2 == 1 )
	//					v( i ) = 0.8*v( 0 )*::sin( 6 * M_PI * v( 0 ) + (i-1)*M_PI/(v.size()-1) );
	//			}
	//
	//
	//
	//			solution.searchPoint() = v;
	//			solution.objectiveFunctionValue() = lz3.eval( v );
	//			solutionSet.push_back( solution );
	//			++it;
	//		}
	//		return( solutionSet );
	//	}
	//
	//
	//	static LZ3::SearchPointType lowerBounds( unsigned int n ) {
	//		LZ3::SearchPointType sp( n, -1. );
	//		sp( 0 ) = 0;
	//
	//		return( sp );
	//	}
	//
	//	static LZ3::SearchPointType upperBounds( unsigned int n ) {
	//		return( LZ3::SearchPointType( n, 1. ) );
	//	}
	//
	//};
}
#endif
