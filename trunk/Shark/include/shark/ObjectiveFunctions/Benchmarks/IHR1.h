//===========================================================================
/*!
* \brief Multi-objective optimization benchmark function IHR 1.
*
*  The function is described in
*
*  Christian Igel, Nikolaus Hansen, and Stefan Roth. 
*  Covariance Matrix Adaptation for Multi-objective Optimization. 
*  Evolutionary Computation 15(1), pp. 1-28, 2007
*
* <BR><HR>
* This file is part of Shark. This library is free software;
* you can redistribute it and/or modify it under the terms of the
* GNU General Public License as published by the Free Software
* Foundation; either version 3, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this library; if not, see <http://www.gnu.org/licenses/>.
*/
//===========================================================================
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_IHR1_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_IHR1_H

#include <shark/ObjectiveFunctions/AbstractMultiObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

#include <shark/LinAlg/rotations.h>

#include <vector>

namespace shark {
/*! \brief Multi-objective optimization benchmark function IHR1.
*
*  The function is described in
*
*  Christian Igel, Nikolaus Hansen, and Stefan Roth. 
*  Covariance Matrix Adaptation for Multi-objective Optimization. 
*  Evolutionary Computation 15(1), pp. 1-28, 2007 
*/
struct IHR1 : public AbstractMultiObjectiveFunction< VectorSpace<double> >
{
	typedef AbstractMultiObjectiveFunction< VectorSpace<double> > super;

	IHR1(std::size_t numVariables = 0) 
	: super( 2 ), m_a( 1000 )
	, m_handler(SearchPointType(numVariables,-1),SearchPointType(numVariables,1) ){
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= IS_CONSTRAINED_FEATURE;
		m_features |= HAS_CONSTRAINT_HANDLER;
		m_features |= CAN_PROVIDE_CLOSEST_FEASIBLE;
		m_name="IHR1";
	}
	
	std::size_t numberOfVariables()const{
		return m_handler.dimensions();
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}

	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_handler.setBounds(
			SearchPointType(numberOfVariables,-1),
			SearchPointType(numberOfVariables,1)
		);
	}
	
	BoxConstraintHandler<SearchPointType> const& getConstraintHandler()const{
		return m_handler;
	}

	void init() {
		m_rotationMatrix = randomRotationMatrix( numberOfVariables() );
	}

	ResultType eval( const SearchPointType & x )const {
		m_evaluationCounter++;

		ResultType value( 2 );

		SearchPointType y = prod(m_rotationMatrix,x);

		value[0] = ::fabs( y( 0 ) );

		double g = 0;
		double ymax = ::fabs( m_rotationMatrix(0, 0) );

		for( unsigned int i = 1; i < numberOfVariables(); i++ )
			ymax = std::max( ::fabs( m_rotationMatrix(0, i) ), ymax );
		ymax = 1. / ymax;

		for (unsigned i = 1; i < numberOfVariables(); i++)
			g += hg( y( i ) );
		g = 9. * g / (numberOfVariables() - 1.) + 1.;

		value[1] = g * hf(1. - ::sqrt( h( y( 0 ), numberOfVariables()) / g ), y( 0 ), ymax );

		return value;
	}

	double h( double x, double n )const {
		return 1 / ( 1 + ::exp( -x / ::sqrt( n ) ) );
	}

	double hf(double x, double y0, double ymax)const {
		if( ::fabs(y0) <= ymax )
			return x;
		return ::fabs( y0 ) + 1.;
	}

	double hg(double x)const {
		return (x*x) / ( ::fabs(x) + 0.1 );
	}
private:
	double m_a;
	BoxConstraintHandler<SearchPointType> m_handler;
	RealMatrix m_rotationMatrix;
};

ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( IHR1, shark::moo::RealValuedObjectiveFunctionFactory );

}
#endif
