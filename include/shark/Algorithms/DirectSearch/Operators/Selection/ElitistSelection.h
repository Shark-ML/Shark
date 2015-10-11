/*!
 *
 *
 * \brief       Elitist Selection Operator suitable for (mu,lambda) and (mu+lambda) selection
 *
 *
 * \author      O.Krause
 * \date        2014
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_ELITIST_SELECTION_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_ELITIST_SELECTION_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/utility/KeyValuePair.h>
#include <vector>
namespace shark {

/// \brief Survival selection to find the next parent set
///
/// Given a set of individuals, selects the mu best performing individuals.
/// The elements are ordered by a double value returned by the Extractor.
template< typename Extractor >
struct ElitistSelection {

	/// \brief Selects individuals from the range of individuals.
	///
	/// \param [in] it Iterator pointing to the first valid parent individual.
	/// \param [in] itE Iterator pointing to the first invalid parent individual.
	/// \param [in] out Iterator pointing to the first valid element of the output range.
	/// \param [in] outE Iterator pointing to the first invalid element of the output range.
	template<typename InIterator,typename OutIterator>
	void operator()(
		InIterator it, InIterator itE,
		OutIterator out,  OutIterator outE
	){
		std::size_t outputSize = std::distance( out, outE );
		std::vector<KeyValuePair<double, InIterator> > results = order(it, itE);
		if(results.size() < outputSize){
			throw SHARKEXCEPTION("[ElitistSelection] Input range must be bigger than output range");
		}
		
		for(std::size_t i = 0; i != outputSize; ++i, ++out){
			*out = *results[i].value;
		}
	}
	
	/// \brief Selects individuals from the range of individuals.
	///
	/// Instead of using an output range, surviving individuals are marked as selected.
	///
	/// \param [in] population The population where individuals are selected from
	/// \param [in] mu number of individuals to select
	template<typename Population>
	void operator()(
		Population& population,std::size_t mu
	){
		SIZE_CHECK(population.size() >= mu);
		typedef typename Population::iterator InIterator;
		std::vector<KeyValuePair<double, InIterator> > results = order(population.begin(),population.end());
		
		for(std::size_t i = 0; i != mu; ++i){
			results[i].value->select()=true;
		}
		for(std::size_t i = mu; i != results.size(); ++i){
			results[i].value->select() = false;
		}
	}
private:
	///Returns a sorted range of pairs indicating, how often every individual won.
	/// The best individuals are in the back of the range.
	template<class InIterator>
	std::vector<KeyValuePair<double, InIterator> > order(InIterator it, InIterator itE){
		std::size_t size = std::distance( it, itE );
		Extractor e;
		std::vector<KeyValuePair<double, InIterator> > individuals(size);
		for(std::size_t i = 0; i != size; ++i){
			individuals[i].key = e(*(it+i));
			individuals[i].value = it+i;
		}
		std::sort( individuals.begin(), individuals.end());
		return individuals;
	}
};

}

#endif
