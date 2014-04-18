//===========================================================================
/*!
 * 
 *
 * \brief       Objective function DTLZ4
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ4_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ4_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

namespace shark {
/**
 * \brief Implements the benchmark function DTLZ4.
 *
 * See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.18.7531&rep=rep1&type=pdf 
 * The benchmark function exposes the following features:
 *	- Scalable w.r.t. the searchspace and w.r.t. the objective space.
 *	- Highly multi-modal.
 */
struct DTLZ4 : public MultiObjectiveFunction
{
	DTLZ4(std::size_t numVariables = 0) : m_objectives(2), m_handler(SearchPointType(numVariables,0),SearchPointType(numVariables,1) ){
		announceConstraintHandler(&m_handler);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "DTLZ4"; }

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

		ResultType value( numberOfObjectives() );

		//~ const double alpha = 10.;
		const double alpha = 100.; //original, but numerically extremely difficult
		
		int k = numberOfVariables() - numberOfObjectives() + 1 ;
		double g = 0.0;
		for( unsigned int i = numberOfVariables() - k; i < numberOfVariables(); i++ )
			g += sqr( x( i ) - 0.5);

		for (unsigned int i = 0; i < numberOfObjectives(); i++) {
			value[i] = 1.0+g;
			for( unsigned int j = 0; j < numberOfObjectives() - i -1; ++j)
				value[i] *= std::cos(std::pow(x( j ),alpha) * M_PI / 2.0);

			if (i > 0)
				value[i] *= std::sin(std::pow(x(numberOfObjectives() - i -1),alpha) * M_PI / 2.0);
		}

		return value;
		return value;
	}
    
private:
	std::size_t m_objectives;
	BoxConstraintHandler<SearchPointType> m_handler;
};

}
#endif
