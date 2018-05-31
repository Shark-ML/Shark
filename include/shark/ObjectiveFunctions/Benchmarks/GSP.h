/*!
 * \brief       GSP benchmark function for multiobjective optimization
 * 
 * \author      O. Krause
 * \date        2015
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_GSP_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_GSP_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/Core/Random.h>

namespace shark {namespace benchmarks{
/// \brief Real-valued benchmark function with two objectives.
/// \ingroup benchmarks
struct GSP : public MultiObjectiveFunction
{
	GSP(std::size_t numVariables=5) : m_handler(SearchPointType(numVariables,0),SearchPointType(numVariables,10000))  {
		announceConstraintHandler(&m_handler);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "GSP"; }

	std::size_t numberOfObjectives()const{
		return 2;
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
			SearchPointType(numberOfVariables,10000)
		);
	}

	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		ResultType value( 2 );
		double alpha = 1. / ( 2. * m_gamma );

		double sum1 = 0., sum2 = 0.;

		for( std::size_t i = 0; i < x.size(); i++ ) {
			sum1 += sqr( x( i ) );
			sum2 += sqr( 1 - x( i ) );
		}

		double alphaN = 1. / ( std::pow( x.size(), alpha ) );

		value[0] = alphaN * std::pow( sum1, alpha );
		value[1] = alphaN * std::pow( sum2, alpha );

		return( value );
	}
private:
	BoxConstraintHandler<SearchPointType> m_handler;
	double m_gamma;
};

}}
#endif
