#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_FONSECA_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_FONSECA_H

//===========================================================================
/*!
 * 
 * \file        Fonseca.h
 *
 * \brief       Bi-objective real-valued benchmark function proposed by Fonseca and Flemming.
 * 
 * 
 *
 * \author      -
 * \date        2011
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

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {

/// \brief Bi-objective real-valued benchmark function proposed by Fonseca and Flemming.
///
/// Fonseca, C. M. and P. J. Fleming (1998). Multiobjective
/// optimization and multiple constraint handling with evolutionary
/// algorithms-Part II: Application example. IEEE Transactions on
/// Systems, Man, and Cybernetics, Part A: Systems and Humans 28(1),
/// 38-47
/// 
/// The default search space dimension is 3, but the function can
/// handle more dimensions.

struct Fonseca : public MultiObjectiveFunction {

	Fonseca(std::size_t numVariables)
	:m_handler(SearchPointType(numVariables,-4),SearchPointType(numVariables,4) ){
		announceConstraintHandler(&m_handler);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Fonseca"; }

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
			SearchPointType(numberOfVariables,-4),
			SearchPointType(numberOfVariables,4)
		);
	}

	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		ResultType value( 2 );

		const double d = 1. / std::sqrt( static_cast<double>( x.size() ) );
		double sum1 = 0., sum2 = 0.;
		for( unsigned int i = 0; i < x.size(); i++ ) {
			sum1 += sqr( x( i ) - d );
			sum2 += sqr( x( i ) + d );
		}

		value[0] = 1-std::exp( - sum1 );
		value[1] = 1-std::exp( - sum2 );

		return value;
	}
private:
	BoxConstraintHandler<SearchPointType> m_handler;
};
}
#endif
