/*!
 *
 *
 * \brief       Roulette-Wheel-Selection based on fitness-rank-based selection probability assignment.
 *
 * The algorithm is described in: James E. Baker. Adaptive Selection
 * Methods for Genetic Algorithms. In John J. Grefenstette (ed.):
 * Proceedings of the 1st International Conference on Genetic
 * Algorithms (ICGA), pp. 101-111, Lawrence Erlbaum Associates, 1985
 *
 *
 *
 * \author      T.Voss
 * \date        2010-2011
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_LINEARRANKING_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_LINEARRANKING_H

#include <shark/Algorithms/DirectSearch/Operators/Selection/RouletteWheelSelection.h>
#include <shark/Core/utility/KeyValuePair.h>
#include <vector>

namespace shark {

/**
 * \brief Implements a fitness-proportional selection scheme for mating selection that
 * scales the fitness values linearly before carrying out the actual selection.
 *
 * The algorithm is described in: James E. Baker. Adaptive Selection
 * Methods for Genetic Algorithms. In John J. Grefenstette (ed.):
 * Proceedings of the 1st International Conference on Genetic
 * Algorithms (ICGA), pp. 101-111, Lawrence Erlbaum Associates, 1985
 *
 */
template<typename Extractor >
struct LinearRankingSelection {
	LinearRankingSelection(){
		etaMax = 1.1;
	}
	/// \brief Selects individualss from the range of parent and offspring individuals.
	///
	/// The operator carries out the following steps:
	///   - Calculate selection probabilities of parent and offspring individualss
	///     according to their rank.
	///   - Carry out roulette wheel selection on the range of parent and
	///     offspring individualss until the output range is filled.
	///
	/// \param [in] individuals Iterator pointing to the first valid individual.
	/// \param [in] individualsE Iterator pointing to the first invalid individual.
	/// \param [in] out Iterator pointing to the first valid element of the output range.
	/// \param [in] outE Iterator pointing to the first invalid element of the output range.
	///
	template<typename InIterator,typename OutIterator> 
	void operator()( 
		InIterator individuals,
		InIterator individualsE,
		OutIterator out,
		OutIterator outE
	) const{
		
		//compute rank of each individual
		std::size_t size = std::distance( individuals, individualsE );
		std::vector<KeyValuePair<double, InIterator> > individualsPerformance(size);

		for( std::size_t i = 0; i != size; ++i, ++individuals ) {
			Extractor e;
			individualsPerformance[i].value = individuals;
			individualsPerformance[i].key = e(*individuals);
		}
		std::sort( individualsPerformance.begin(), individualsPerformance.end());
		
		RealVector selectionProbability(size);
		double a = 2. * (etaMax - 1.)/(size - 1.);
		for( std::size_t i = 0; i != size; ++i ) {
			selectionProbability[i] = (etaMax - a*i);
		}
		selectionProbability /=sum(selectionProbability);

		RouletteWheelSelection rws;
		for( ; out != outE; ++out ){
			InIterator individuals = rws( individualsPerformance.begin(), individualsPerformance.end(), selectionProbability)->value;
			*out = *individuals;
		}
	}
	
	/// \brief Selective pressure, parameter in [1,2] conrolling selection strength. 1.1 by default
	double etaMax;

};
}

#endif
