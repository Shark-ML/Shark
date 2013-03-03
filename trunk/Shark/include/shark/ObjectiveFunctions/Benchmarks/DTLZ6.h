//===========================================================================
/*!
*  \brief Objective function DTLZ6
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ6_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ6_H

#include <shark/ObjectiveFunctions/AbstractMultiObjectiveFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

namespace shark {
/**
* \brief Implements the benchmark function DTLZ6.
*
* See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.18.7531&rep=rep1&type=pdf
* The benchmark function exposes the following features:
*	- Scalable w.r.t. the searchspace and w.r.t. the objective space.
*	- Highly multi-modal.
*/
struct DTLZ6 : public AbstractMultiObjectiveFunction< VectorSpace<double> >
{
	typedef AbstractMultiObjectiveFunction< VectorSpace<double> > super;
	
	DTLZ6(std::size_t numVariables = 0) : super( 2 ),m_handler(SearchPointType(numVariables,0),SearchPointType(numVariables,1) ){
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= IS_CONSTRAINED_FEATURE;
		m_features |= HAS_CONSTRAINT_HANDLER;
		m_features |= CAN_PROVIDE_CLOSEST_FEASIBLE;
		m_name="DTLZ6";
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

		ResultType value( noObjectives() );

		std::vector<double> phi(noObjectives());

		int k = numberOfVariables() - noObjectives() + 1 ;
		double g = 0.0 ;

		for (unsigned int i = numberOfVariables() - k + 1; i <= numberOfVariables(); i++)
			g += std::pow(x(i-1), 0.1);

		double t = M_PI  / (4 * (1 + g));

		phi[0] = x(0) * M_PI / 2;
		for (unsigned int i = 2; i <= noObjectives() - 1; i++)
			phi[i-1] = t * (1 + 2 * g * x(i-1) );

		for (unsigned int i = 1; i <= noObjectives(); i++)
		{
			double f = (1 + g);

			for (int j = noObjectives() - i; j >= 1; j--)
				f *= std::cos(phi[j-1]);

			if (i > 1)
				f *= std::sin(phi[(noObjectives() - i + 1) - 1]);

			value[i-1] = f ;
		}

		return( value );
	}

private:
	BoxConstraintHandler<SearchPointType> m_handler;
};

//~ /**
 //~ * \brief Specializes MultiObjectiveFunctionTraits for DTLZ6.
 //~ */
//~ template<> struct MultiObjectiveFunctionTraits<DTLZ6> {

	//~ /**
	//~ * \brief Models the reference Pareto-front type.
	//~ */
	//~ typedef std::vector< DTLZ6::ResultType > ParetoFrontType;

	//~ /**
	//~ * \brief Models the reference Pareto-set type.
	//~ */
	//~ typedef std::vector< DTLZ6::SearchPointType > ParetoSetType;

//~ };

	ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( DTLZ6, shark::moo::RealValuedObjectiveFunctionFactory );
}
#endif
