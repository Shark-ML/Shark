/*!
 *
 *
 * \brief       Roulette-Wheel-Selection using uniform selection probability assignment.
 *
 *
 *
 * \author      T.Voss
 * \date        2010-2011
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_UNIFORMRANKING_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_UNIFORMRANKING_H

#include <shark/Algorithms/DirectSearch/Operators/Selection/RouletteWheelSelection.h>

namespace shark {
/// \brief Selects individuals from the range of individual and offspring individuals.
struct UniformRankingSelection {

	/// \brief Selects individuals from the range of individual and offspring individuals.
	///
	/// The operator carries out the following steps:
	///   - Assign uniform selection probabilities to all individuals.
	///   - Carry out roulette wheel selection on the range of individual and 
	///     offspring individuals until the output range is filled.
	///
	/// \param [in] individuals Iterator pointing to the first valid individual.
	/// \param [in] individualsE Iterator pointing to the first invalid individual.
	/// \param [in] out Iterator pointing to the first valid element of the output range.
	/// \param [in] outE Iterator pointing to the first invalid element of the output range.
	template<typename InIterator,typename OutIterator> 
	void operator()( 
		InIterator individuals,
		InIterator individualsE,
		OutIterator out,
		OutIterator outE
	){
		std::size_t size = std::distance( individuals, individualsE );
		
		RealVector selectionProbability(size,1.0/size);
		RouletteWheelSelection rws;
		for( ; out != outE; ++out ){
			*out = *rws( individuals, individualsE, selectionProbability);
		}
	}

};

}

#endif
