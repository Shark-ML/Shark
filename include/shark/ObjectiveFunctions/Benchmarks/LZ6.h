//===========================================================================
/*!
 * 
 *
 * \brief       Multi-objective optimization benchmark function LZ6.
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_LZ6_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_LZ6_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

namespace shark {
/*! \brief Multi-objective optimization benchmark function LZ6.
*
*  The function is described in
*
*  H. Li and Q. Zhang. 
*  Multiobjective Optimization Problems with Complicated Pareto Sets, MOEA/D and NSGA-II, 
*  IEEE Trans on Evolutionary Computation, 2(12):284-302, April 2009. 
*/
struct LZ6 : public MultiObjectiveFunction
{
	LZ6(std::size_t numVariables = 0) : m_handler(SearchPointType(numVariables,-2),SearchPointType(numVariables,2) ){
		announceConstraintHandler(&m_handler);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LZ6"; }
	
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
		SearchPointType lb(numberOfVariables,-2);
		SearchPointType ub(numberOfVariables, 2);
		lb(0) = 0;
		ub(0) = 1;
		m_handler.setBounds(lb, ub);
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
private:
	BoxConstraintHandler<SearchPointType> m_handler;
};

}
#endif
