//===========================================================================
/*!
 * 
 *
 * \brief       Summarizes definitions and tags common to the EA/DirectSearch component.

 * 
 *
 * \author      T.Voss
 * \date        2010
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
