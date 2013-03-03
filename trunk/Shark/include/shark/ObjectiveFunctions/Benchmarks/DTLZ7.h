//===========================================================================
/*!
*  \brief Objective function DTLZ7
*
*  \author T.Voss, T. Glasmachers, O.Krause
*  \date 2010-2011
*
*  \par Copyright (c) 1998-2007:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
*
*/
//===========================================================================
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ7_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ7_H

#include <shark/ObjectiveFunctions/AbstractMultiObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

namespace shark {
/**
* \brief Implements the benchmark function DTLZ7.
*
* See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.18.7531&rep=rep1&type=pdf
* The benchmark function exposes the following features:
*	- Scalable w.r.t. the searchspace and w.r.t. the objective space.
*	- Disconnected Pareto front.
*/
struct DTLZ7 : public AbstractMultiObjectiveFunction< VectorSpace<double> >
{
	typedef AbstractMultiObjectiveFunction< VectorSpace<double> > super;
	
	DTLZ7(std::size_t numVariables = 0) : super( 2 ),m_handler(SearchPointType(numVariables,0),SearchPointType(numVariables,1) ){
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= IS_CONSTRAINED_FEATURE;
		m_features |= HAS_CONSTRAINT_HANDLER;
		m_features |= CAN_PROVIDE_CLOSEST_FEASIBLE;
		m_name="DTLZ7";
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
	
	BoxConstraintHandler<SearchPointType> const& getConstraintHandler()const{
		return m_handler;
	}

	ResultType eval( const SearchPointType & x ) const {
		m_evaluationCounter++;

		RealVector value( noObjectives() );

		int k = numberOfVariables() - noObjectives() + 1 ;
		double g = 0.0 ;
		for (unsigned int i = numberOfVariables() - k + 1; i <= numberOfVariables(); i++)
			g += x(i-1);

		g = 1 + 9 * g / k;

		for (unsigned int i = 1; i <= noObjectives() - 1; i++)
			value[i-1] = x(i-1);

		double h = 0.0 ;
		for (unsigned int j = 1; j <= noObjectives() - 1; j++)
			h += x(j-1) / (1 + g) * ( 1 + std::sin( 3 * M_PI * x(j-1) ) );

		h = noObjectives() - h ;

		value[noObjectives()-1] = (1 + g) * h;

		return value;
	}

private:
	BoxConstraintHandler<SearchPointType> m_handler;
};

	ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( DTLZ7, shark::moo::RealValuedObjectiveFunctionFactory );
}
#endif
