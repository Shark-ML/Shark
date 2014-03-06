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

#include <shark/Algorithms/DirectSearch/EA.h>
#include <shark/Algorithms/DirectSearch/FitnessComparator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/UniformRanking.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/RouletteWheelSelection.h>

namespace shark {

/**
 * \brief Implements a fitness-proportional selection scheme that
 * scales the fitness values linearly before carrying out the actual selection.
 *
 * The algorithm is described in: James E. Baker. Adaptive Selection
 * Methods for Genetic Algorithms. In John J. Grefenstette (ed.):
 * Proceedings of the 1st International Conference on Genetic
 * Algorithms (ICGA), pp. 101-111, Lawrence Erlbaum Associates, 1985
 *
 */
template< typename FitnessType >
struct LinearRankingSelection {

	typedef FitnessType fitness_type;

	/**
	 * \brief Selects individuals from the range of parent and offspring individuals.
	 *
	 * The operator carries out the following steps:
	 *   - Calculate selection probabilities of parent and offspring individuals
	 *     according to their rank.
	 *   - Carry out roulette wheel selection on the range of parent and
	 *     offspring individuals until the output range is filled.
	 *
	 * \param [in] parents Iterator pointing to the first valid parent individual.
	 * \param [in] parentsE Iterator pointing to the first invalid parent individual.
	 * \param [in] offspring Iterator pointing to the first valid offspring individual.
	 * \param [in] offspringE Iterator pointing to the first invalid offspring individual.
	 * \param [in] out Iterator pointing to the first valid element of the output range.
	 * \param [in] outE Iterator pointing to the first invalid element of the output range.
	 * \param [in] etaMax Selective pressure, parameter in [1,2] conrolling selection strength.
	 *
	 */
	template<
	    typename InIterator,
	    typename OutIterator
	    > void operator()( InIterator parents,
	                       InIterator parentsE,
	                       InIterator offspring,
	                       InIterator offspringE,
	                       OutIterator out,
	                       OutIterator outE,
	                       double etaMax
	                     ) const
	{

		std::size_t noParents = std::distance( parents, parentsE );
		std::size_t noOffspring = std::distance( offspring, offspringE );
		std::size_t totalSize = noParents + noOffspring;
		//Assumption: Both parent and offspring ranges are sorted.
		const double a = 2. * (etaMax - 1.)/(totalSize - 1.);

		std::vector< InIterator > view( totalSize );
		typename std::vector< InIterator >::iterator itv = view.begin();

		for( InIterator it = parents; it != parentsE; ++it, ++itv ) {
			*itv = it;
		}

		for( InIterator it = offspring; it != offspringE; ++it, ++itv ) {
			*itv = it;
		}

		std::sort( view.begin(), view.end(), shark::IndirectFitnessComparator< FitnessType >() );

		std::size_t counter = 0;
		for( typename std::vector< InIterator >::iterator it = view.begin(); it != view.end(); ++it, counter++ ) {
			(*it)->probability( shark::tag::SelectionProbability() ) = (etaMax - a*counter)/static_cast<double>( totalSize );
		}

		RouletteWheelSelection rws;
		detail::IteratorSelectionProbabilityExtractor<
		shark::tag::SelectionProbability
		> ext;

		for( ; out != outE; ++out )
			*out = **rws( view.begin(), view.end(), ext );
	}

	/**
	 * \brief Selects individuals from the range of parent and offspring individuals.
	 * \param [in] parents Range of parent individuals.
	 * \param [in] offspring Range of offspring individuals.
	 * \param [in] out Output range.
	 * \param [in] etaMax Selective pressure, parameter in [1,2] conrolling selection strength.
	 *
	 */
	template<
	    typename InRange,
	    typename OutRange
	    > void operator()( InRange parents,
	                       InRange offspring,
	                       OutRange out,
	                       double etaMax )
	{
		(*this)( parents.begin(), parents.end(), offspring.begin(), offspring.end(), out.begin(), out.end(), etaMax );
	}

};
}

#endif
