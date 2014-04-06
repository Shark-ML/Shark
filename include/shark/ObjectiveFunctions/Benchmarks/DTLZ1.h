//===========================================================================
/*!
 * 
 *
 * \brief       Objective function DTLZ1
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ1_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ1_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

namespace shark {

/**
* \brief Implements the benchmark function DTLZ1.
*
* See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.18.7531&rep=rep1&type=pdf
* The benchmark function exposes the following features:
*	- Scalable w.r.t. the searchspace and w.r.t. the objective space.
*	- Highly multi-modal.
*/
struct DTLZ1 : public MultiObjectiveFunction
{
	DTLZ1(std::size_t numVariables = 0) : m_objectives(2), m_handler(SearchPointType(numVariables,0),SearchPointType(numVariables,1) ){
		announceConstraintHandler(&m_handler);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "DTLZ1"; }

	std::size_t numberOfObjectives()const{
		return m_objectives;
	}
	bool hasScalableObjectives()const{
		return true;
	}
	
	void setNumberOfObjectives( std::size_t numberOfObjectives ){
		m_objectives = numberOfObjectives;
	}
	
	
	std::size_t numberOfVariables()const{
		return m_handler.dimensions();
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}

	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_handler.setBounds(
			SearchPointType(numberOfVariables,0),
			SearchPointType(numberOfVariables,1)
		);
	}

	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		ResultType value( numberOfObjectives() );

		int k = numberOfVariables() - numberOfObjectives() + 1 ;
		// TODO: Check k
		double g = 0.0;

		for( unsigned int i = numberOfVariables() - k + 1; i <= numberOfVariables(); i++ )
		    g += sqr( x( i-1 ) - 0.5 ) - std::cos( 20 * M_PI * ( x( i-1 ) - 0.5) );

		g = 100 * (k + g);

		for (unsigned int i = 1; i <= numberOfObjectives(); i++) {
			double f = 0.5 * (1 + g);
			for( unsigned int j = numberOfObjectives() - i; j >= 1; j--)
				f *= x( j-1 );

			if (i > 1)
				f *= 1 - x( (numberOfObjectives() - i + 1) - 1);

			value[i-1] = f;
		}

		return value;
	}
private:
	std::size_t m_objectives;
	BoxConstraintHandler<SearchPointType> m_handler;
	
};

}

#endif
