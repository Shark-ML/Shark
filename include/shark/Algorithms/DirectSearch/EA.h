//===========================================================================
/*!
*
*  \brief Summarizes definitions and tags common to the EA/DirectSearch component.
*
*  \author T.Voss
*  \date 2010
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

#ifndef SHARK_EA_H
#define SHARK_EA_H

namespace shark {

	namespace tag {

		/** \brief Marks penalized fitness values. */
		struct PenalizedFitness {};
		
		/** \brief Marks unpenalized fitness values. */
		struct UnpenalizedFitness {};

		/** \brief Marks scaled fitness values. */
		struct ScaledFitness {};
        
		/** \brief Marks the mating probability of an individual. */
		struct MatingProbability {};

		/** \brief Marks the selection probability of an individual. */
		struct SelectionProbability {};
	}
}

#endif // SHARK_EA_H
