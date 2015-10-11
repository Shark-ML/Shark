//===========================================================================
/*!
 * 
 *
 * \brief       Multi-objective optimization benchmark function ZDT6
 * 
 * The function is described in
 * 
 * Eckart Zitzler, Kalyanmoy Deb, and Lothar Thiele. Comparison of
 * Multiobjective Evolutionary Algorithms: Empirical
 * Results. Evolutionary Computation 8(2):173-195, 2000
 * 
 * 
 *
 * \author      -
 * \date        -
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_ZDT6_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_ZDT6_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

namespace shark {
/*! \brief Multi-objective optimization benchmark function ZDT6
*
*  The function is described in
*
*  Eckart Zitzler, Kalyanmoy Deb, and Lothar Thiele. Comparison of
*  Multiobjective Evolutionary Algorithms: Empirical
*  Results. Evolutionary Computation 8(2):173-195, 2000
*/
struct ZDT6 : public MultiObjectiveFunction
{
	
	ZDT6(std::size_t numVariables = 0) : m_handler(numVariables,0,1){
		announceConstraintHandler(&m_handler);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ZDT6"; }

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
		m_handler.setBounds(numberOfVariables,0,1);
	}

	// std::vector<double> evaluate( const point_type & x ) {
	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		ResultType value( 2 );

		value[0] = 1.0 - std::exp(-4.0 * x( 0 )) * std::pow( std::sin(6 * M_PI * x( 0 ) ), 6);

		double mean = sum(x) - x(0);
		mean /= (numberOfVariables() - 1.0);

		double g = 1.0 + 9.0 * std::pow(mean, 0.25);
		double h = 1.0 - sqr(value[0] / g);
		value[1] = g*h;

		return value;
	}
private:
	BoxConstraintHandler<SearchPointType> m_handler;
};

}
#endif
