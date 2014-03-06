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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_UNIFORMRANKING_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_UNIFORMRANKING_H

#include <shark/Algorithms/DirectSearch/EA.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/RouletteWheelSelection.h>

namespace shark {

namespace detail {

/** \cond */
template <typename SelectionProbabilityType>
struct IteratorSelectionProbabilityExtractor {

	typedef SelectionProbabilityType selection_probability_type;

	template<typename Iterator>
	inline double operator()( Iterator it ) const
	{
		return( it->probability( selection_probability_type() ) );
	}

};
/** \endcond */
}

/**
 * \brief Selects individuals from the range of parent and offspring individuals.
 */
template< typename FitnessType >
struct UniformRankingSelection {

	/** \brief Marks the user selected fitness type. */
	typedef FitnessType fitness_type;

	/**
	 * \brief Selects individuals from the range of parent and offspring individuals.
	 *
	 * The operator carries out the following steps:
	 *   - Assign uniform selection probabilities to parent and offspring individuals.
	 *   - Carry out roulette wheel selection on the range of parent and offspring individuals until the output range is filled.

	 * \param [in] parents Iterator pointing to the first valid parent individual.
	 * \param [in] parentsE Iterator pointing to the first invalid parent individual.
	 * \param [in] offspring Iterator pointing to the first valid offspring individual.
	 * \param [in] offspringE Iterator pointing to the first invalid offspring individual.
	 * \param [in] out Iterator pointing to the first valid element of the output range.
	 * \param [in] outE Iterator pointing to the first invalid element of the output range.
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
	                       OutIterator outE )
	{

		std::size_t noParents = std::distance( parents, parentsE );
		std::size_t noOffspring = std::distance( offspring, offspringE );
		std::size_t totalSize = noParents + noOffspring;
		//Assumption: Both parent and offspring ranges are sorted.
		const double p = 1./totalSize;

		std::vector< InIterator > view( totalSize );
		typename std::vector< InIterator >::iterator itv = view.begin();

		std::size_t counter = 0;
		for( InIterator it = parents; it != parentsE; ++it, ++counter, ++itv ) {
			it->probability( shark::tag::SelectionProbability() ) = p;
			*itv = it;
		}

		for( InIterator it = offspring; it != offspringE; ++it, ++counter, ++itv ) {
			it->probability( shark::tag::SelectionProbability() ) = p;
			*itv = it;
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
	 *
	 */
	template<
	    typename InRange,
	    typename OutRange
	    > void operator()( InRange parents,
	                       InRange offspring,
	                       OutRange out )
	{
		(*this)( parents.begin(), parents.end(), offspring.begin(), offspring.end(), out.begin(), out.end() );
	}

};

}

#endif
