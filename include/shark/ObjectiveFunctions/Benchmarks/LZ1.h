//===========================================================================
/*!
 * 
 * \file        LZ1.h
 *
 * \brief       Multi-objective optimization benchmark function LZ1.
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_LZ1_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_LZ1_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

namespace shark {
/*! \brief Multi-objective optimization benchmark function LZ1.
*
*  The function is described in
*
* H. Li and Q. Zhang. 
* Multiobjective Optimization Problems with Complicated Pareto Sets, MOEA/D and NSGA-II, 
* IEEE Trans on Evolutionary Computation, 2(12):284-302, April 2009. 
*/
struct LZ1 : public MultiObjectiveFunction
{
	LZ1(std::size_t numVariables = 0) : m_handler(SearchPointType(numVariables,0),SearchPointType(numVariables,1) ){
		announceConstraintHandler(&m_handler);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LZ1"; }

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
		m_handler.setBounds(
			SearchPointType(numberOfVariables,0),
			SearchPointType(numberOfVariables,1)
		);
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
private:
	BoxConstraintHandler<SearchPointType> m_handler;
};

ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( LZ1, shark::moo::RealValuedObjectiveFunctionFactory );
//template<> struct ObjectiveFunctionTraits<LZ1> {
	//	static LZ1::SolutionSetType referenceSet( std::size_t maxSize,
	//		unsigned int numberOfVariables,
	//		unsigned int numberOfObjectives ) {
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
