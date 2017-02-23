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
 * \par Copyright 1995-2017 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#include <vector>
#include <numeric>
namespace shark {

/// \brief Survival selection to find the next parent set
///
/// Given a set of individuals, selects the mu best performing individuals.
/// The elements are ordered using the given Ordering Relation
template< typename Ordering >
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
		std::vector<InIterator> results = order(it, itE);
		SHARK_RUNTIME_CHECK(results.size() > outputSize, "Input range must be bigger than output range");
		
		for(std::size_t i = 0; i != outputSize; ++i, ++out){
			*out = *results[i];
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
		SHARK_RUNTIME_CHECK(population.size() >= mu, "Population Size must be at least mu");

		typedef typename Population::iterator InIterator;
		std::vector<InIterator> results = order(population.begin(),population.end());
		
		for(std::size_t i = 0; i != mu; ++i){
			results[i]->select()=true;
		}
		for(std::size_t i = mu; i != results.size(); ++i){
			results[i]->select() = false;
		}
	}
private:
	/// Returns a sorted range of pairs indicating, how often every individual won.
	/// The best individuals are in the back of the range.
	template<class InIterator>
	std::vector<InIterator> order(InIterator it, InIterator itE){
		std::size_t size = std::distance( it, itE );
		std::vector<InIterator > individuals(size);
		std::iota(individuals.begin(),individuals.end(),it);
		std::sort(
			individuals.begin(),
			individuals.end(),
			[](InIterator lhs, InIterator rhs){
				Ordering ordering;
				return ordering(*lhs,*rhs);
			}
		);
		return individuals;
	}
};

}

#endif
