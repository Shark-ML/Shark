//===========================================================================
/*!
 * 
 *
 * \brief       Objective function DTLZ7
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ7_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ7_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

namespace shark {
/**
* \brief Implements the benchmark function DTLZ7.
*
* See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.18.7531&rep=rep1&type=pdf
* The benchmark function exposes the following features:
*	- Scalable w.r.t. the searchspace and w.r.t. the objective space.
*	- Disconnected Pareto front.
*/
struct DTLZ7 : public MultiObjectiveFunction
{
	DTLZ7(std::size_t numVariables = 0) : m_objectives(2), m_handler(numVariables,0,1 ){
		announceConstraintHandler(&m_handler);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "DTLZ7"; }

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

		RealVector value( numberOfObjectives() );

		int k = numberOfVariables() - numberOfObjectives() + 1 ;
		double g = 0.0 ;
		for (unsigned int i = numberOfVariables() - k + 1; i <= numberOfVariables(); i++)
			g += x(i-1);

		g = 1 + 9 * g / k;

		for (unsigned int i = 0; i != numberOfObjectives(); i++)
			value[i] = x(i);

		double h = 0.0 ;
		for (unsigned int j = 1; j <= numberOfObjectives() - 1; j++)
			h += x(j-1) / (1 + g) * ( 1 + std::sin( 3 * M_PI * x(j-1) ) );

		h = numberOfObjectives() - h ;

		value[numberOfObjectives()-1] = (1 + g) * h;

		return value;
	}

private:
	std::size_t m_objectives;
	BoxConstraintHandler<SearchPointType> m_handler;
	
};

}
#endif
