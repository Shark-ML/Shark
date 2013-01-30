//===========================================================================
/*!
*  \file ObjectiveFunctionTraits.h
*
*  \brief ObjectiveFunctionTraits
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
#ifndef SHARK_CORE_TRAITS_OBJECTIVEFUNCTIONTRAITS_H
#define SHARK_CORE_TRAITS_OBJECTIVEFUNCTIONTRAITS_H

#include <limits>

namespace shark {

	/**
	* \brief Abstract traits specific to an objective function type that 
	* are not modeled in interface AbstractObjectiveFunction.
	*
	* The traits defined here contain accessors for the box constraints of
	* an objective function. In order to integrate a new type of objective function
	* with the library, provide a template specialization for the respective objective
	* function type, e.g.:
	*	\code
	*	template<> struct ObjectiveFunctionTraits<DTLZ1> {
	*
	*		static DTLZ1::SearchPointType lowerBounds( unsigned int n ) {
	*			return DTLZ1::SearchPointType( n, 0. );
	*		}
	*
	*		static DTLZ1::SearchPointType upperBounds( unsigned int n ) {
	*			return DTLZ1::SearchPointType( n, 1. );
	*		}
	*
	*	};
	*	\endcode
	*/
	template<typename ObjectiveFunction>
	struct ObjectiveFunctionTraits {

		/**
		* \brief Queries the number of variables of a function.
		* \param [in] f The function to query.
		*/
		static unsigned int numberOfVariables( const ObjectiveFunction & f ) {
			return( f.numberOfVariables() );
		}

		/**
		* \brief Adjusts the number of variables of a function.
		* \param [in,out] f The function to adjust the number of variables for.
		* \param [in] n The new number of variables.
		*/
		static unsigned int setNumberOfVariables( ObjectiveFunction & f, unsigned int n ) {
			return( f.setNumberOfVariables( n ) );
		}

		/**
		* \brief Returns the lower bounds of a box constrained objective function type.
		* \param [in] n The dimensionality of the search space.
		* \returns A point p modelling the lower bounds with p.size() <= n.
		*/
		static typename ObjectiveFunction::SearchPointType lowerBounds( unsigned int n ) {
			return( 
				typename ObjectiveFunction::SearchPointType( 
					n, 
					0.
				) 
			);
		}

		/**
		* \brief Returns the upper bounds of a box constrained objective function type.
		* \param [in] n The dimensionality of the search space.
		* \returns A point p modelling the upper bounds with p.size() <= n.
		*/
		static typename ObjectiveFunction::SearchPointType upperBounds( unsigned int n ) {
			return( 
				typename ObjectiveFunction::SearchPointType( 
					n, 
					1.
				) 
			);
		}
	};
}

#endif
