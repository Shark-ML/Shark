//===========================================================================
/*!
 * 
 *
 * \brief       Multi-objective optimization benchmark function IHR 4.
 * 
 * The function is described in
 * 
 * Christian Igel, Nikolaus Hansen, and Stefan Roth. 
 * Covariance Matrix Adaptation for Multi-objective Optimization. 
 * Evolutionary Computation 15(1), pp. 1-28, 2007
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_IHR4_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_IHR4_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

#include <shark/LinAlg/rotations.h>

namespace shark{
/*! \brief Multi-objective optimization benchmark function IHR 4.
*
*  The function is described in
*
*  Christian Igel, Nikolaus Hansen, and Stefan Roth. 
*  Covariance Matrix Adaptation for Multi-objective Optimization. 
*  Evolutionary Computation 15(1), pp. 1-28, 2007 
*/
struct IHR4 : public MultiObjectiveFunction
{
	IHR4(std::size_t numVariables = 0) 
	: m_a( 1000 )
	, m_handler(SearchPointType(numVariables,-5),SearchPointType(numVariables,5) ){
		announceConstraintHandler(&m_handler);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "IHR4"; }
	
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
			SearchPointType(numberOfVariables,-5),
			SearchPointType(numberOfVariables,5)
		);
	}

	void init() {
		m_rotationMatrix = blas::randomRotationMatrix(numberOfVariables());
	}

	ResultType eval( const SearchPointType & x )const {
		m_evaluationCounter++;

		ResultType value( 2 );

		SearchPointType y = prod(m_rotationMatrix,x);

		value[0] = std::abs( y( 0 ) );

		double g = 0;
		double ymax = std::abs( m_rotationMatrix(0, 0) );

		for( unsigned int i = 1; i < numberOfVariables(); i++ )
			ymax = std::max( std::abs( m_rotationMatrix(0, i) ), ymax );
		ymax = 1. / ymax;

		for (unsigned i = 1; i < numberOfVariables(); i++)
			g += sqr( y( i ) ) - 10 * std::cos( 4 * M_PI * y( i ) ); //hg( y( i ) );
		g += 10 * (numberOfVariables() - 1.) + 1.;

		value[1] = g * hf(1. -std::sqrt( h( y( 0 ), numberOfVariables()) / g ), y( 0 ), ymax );

		return value;
	}

	double h( double x, double n )const {
		return 1 / ( 1 +std::exp( -x / std::sqrt( n ) ) );
	}

	double hf(double x, double y0, double ymax)const {
		if( std::abs(y0) <= ymax )
			return x;
		return std::abs( y0 ) + 1.;
	}

	double hg(double x) {
		return (x*x) / ( std::abs(x) + 0.1 );
	}
private:
	double m_a;
	BoxConstraintHandler<SearchPointType> m_handler;
	RealMatrix m_rotationMatrix;
};

}
#endif // IHR1_H
