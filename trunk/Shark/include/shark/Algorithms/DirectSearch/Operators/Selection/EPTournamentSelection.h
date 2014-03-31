/*!
 *
 *
 * \brief       EP-Tournament selection operator.
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_EP_TOURNAMENT_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_SELECTION_EP_TOURNAMENT_H

#include <shark/LinAlg/Base.h>
#include <shark/LinAlg/Core/KeyValuePair.h>
#include <vector>
namespace shark {

/// \brief Survival and mating selection to find the next parent set.
///
/// For a given Tournament size k, every individual is compared to k other individuals
/// The fitness relation is governed by the double value returned by Extractor, which can be the fitness or a 
/// domination rank. The individuals which won the most torunaments are selected
template< typename Extractor >
struct EPTournamentSelection {

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
		std::vector<KeyValuePair<unsigned int, InIterator> > results = performTournament(it, itE);
		SIZE_CHECK(results.size() >= outputSize);
		
		for(std::size_t i = 0; i != outputSize; ++i, ++out){
			*out = *results[size-i-1].value;
		}
	}
	
	/// \brief Selects individuals from the range of individuals.
	///
	/// Instead of using an output range, surviving individuals are marked as selected.
	///
	/// \param [in] it Iterator pointing to the first valid parent individual.
	/// \param [in] itE Iterator pointing to the first invalid parent individual.
	/// \param [in] mu number of individuals to select
	template<typename Range>
	void operator()(
		Population& population,std::size_t mu
	){
		SIZE_CHECK(population.size() >= mu);
		typedef typename Population::iterator InIterator;
		std::vector<KeyValuePair<unsigned int, InIterator> > results = performTournament(population.begin(),population.end());
		
		
		for(std::size_t i = 0; i != mu; ++i, ++out){
			individualPerform[size-i-1].value->select()=true;
		}
		for(std::size_t i = mu; i != results.size(); ++i, ++out){
			individualPerform[size-i-1].value->select() = false;
		}
	}
	
	/// \brief Size of the tournament. 4 by default.
	std::size_t tournamentSize;
private:
	///Returns a sorted range of pairs indicating, how often every individual won.
	/// The best individuals are in the back of the range.
	std::vector<KeyValuePair<unsigned int, InIterator> > performTournament(InIterator it, InIterator itE){
		std::size_t size = std::distance( it, itE );
		UIntVector selectionProbability(size,0.0);
		std::vector<KeyValuePair<unsigned int, InIterator> > individualPerformance(size);
		Extractor e;
		for( std::size_t i = 0; i != size(); ++i ) {
			individualPerformance[i].value = it+i;
			for( std::size_t round = 0; round < tournamentSize; round++ ) {
				std::size_t idx = shark::Rng::discrete( 0,size-1 );
				if(e(*it) < e(*(it+idx)){
					individualPerformance[i].key += 1;
				}
			}
		}
		
		std::sort( individualPerformance.begin(), individualPerformance.end());
		return individualPerformance;
	}
};

}

#endif
