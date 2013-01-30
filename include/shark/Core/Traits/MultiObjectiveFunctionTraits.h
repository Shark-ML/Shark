/**
*
*  \brief MultiObjectiveFunctionTraits
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
#ifndef SHARK_CORE_TRAITS_MULTI_OBJECTIVE_FUNCTION_TRAITS_H
#define SHARK_CORE_TRAITS_MULTI_OBJECTIVE_FUNCTION_TRAITS_H

#include <limits>

namespace shark {

	/**
	* \brief Abstract traits specific to a multi-objective function type that 
	* are not modeled in interface AbstractObjectiveFunction.
	*
	* The traits defined here allow for generating reference Pareto fronts and sets.
	*/
	template<typename ObjectiveFunction>
	struct MultiObjectiveFunctionTraits {

		/**
		* \brief Models the reference Pareto-front type.
		*/
		typedef std::vector< typename ObjectiveFunction::ResultType > ParetoFrontType;

		/**
		* \brief Models the reference Pareto-set type.
		*/
		typedef std::vector< typename ObjectiveFunction::SearchPointType > ParetoSetType;

		/**
		* \brief Returns the reference Pareto front of the objective function type.
		* \param [in] noPoints The number of points in the reference Pareto front.
		* \param [in] n The dimensionality of the reference Pareto set
		* \param [in] m The number of objectives
		* \returns The reference front of the problem or an empty set.
		*/
		static std::vector< typename ObjectiveFunction::ResultType > referenceFront( std::size_t noPoints, std::size_t n, std::size_t m ) {
			return( std::vector< typename ObjectiveFunction::ResultType >() );
		}

		/**
		* \brief Returns the reference Pareto set of the objective function type.
		* \param [in] noPoints The number of points in the reference Pareto set.
		* \param [in] n The dimensionality of the reference Pareto set
		* \param [in] m The number of objectives
		* \returns The reference set of the problem or an empty set.
		*/
		static std::vector< typename ObjectiveFunction::SearchPointType > referenceSet( std::size_t noPoints, std::size_t n, std::size_t m ) {
			return( std::vector< typename ObjectiveFunction::SearchPointType >() );
		}
	};
}

#endif