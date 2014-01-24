/*!
 * 
 * \file        RouletteWheelSelection.h
 *
 * \brief       Implements fitness proportional selection.
 * 
 * See http://en.wikipedia.org/wiki/Fitness_proportionate_selection
 * 
 * 
 *
 * \author      T. Voss
*  \par Copyright (c) 1998-2008:
*      Institut f&uuml;r Neuroinformatik
 * \date        -
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_ROULETTE_WHEEL_SELECTION_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_ROULETTE_WHEEL_SELECTION_H

#include <shark/Rng/GlobalRng.h>

#include <numeric>
#include <vector>

namespace shark {
/**
* \brief Fitness-proportional selection operator.
*
* See http://en.wikipedia.org/wiki/Fitness_proportionate_selection.
*/
struct RouletteWheelSelection {
	/**
	* \brief Selects an individual from the range of individuals with prob. proportional to its fitness.
	* \tparam Extractor Type for mapping elements from the underlying range on the real line.
	* \param [in] it Iterator pointing to the first valid element.
	* \param [in] itE Iterator pointing to the first invalid element.
	* \param [in, out] e Object of type Extractor for extracting the elements selection probability.
	* \returns An iterator pointing to the selected individual.
	*/
	template<typename Iterator, typename Extractor>
	Iterator operator()(Iterator it, Iterator itE, Extractor &e) const {
		std::size_t n = std::distance(it, itE);

		std::vector< double > rouletteWheel(n, 0.);

		std::transform(it,
		        itE,
		        rouletteWheel.begin(),
		        e
		);

		std::partial_sum(rouletteWheel.begin(), rouletteWheel.end(), rouletteWheel.begin());

		double rnd = Rng::uni();
		Iterator result;
		std::vector< double >::iterator itv = rouletteWheel.begin();
		for (result = it; result != itE; ++result, ++itv) {
			if (rnd <= *itv)
				break;
		}

		return(result);
	}
};
}

#endif