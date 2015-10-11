/*!
 * 
 *
 * \brief       Bit flip mutation operator.
 * 
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_MUTATION_BITFLIP_MUTATION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_MUTATION_BITFLIP_MUTATION_H

#include <shark/Rng/GlobalRng.h>

namespace shark {

/// \brief Bitflip mutation operator.
///
/// Given a binary vector, a coin is flipped for every element. If it is heads, the element of the vector
/// is flipped. By initializing the mutator with the objective function, this strength is set to a dfault value
/// of 1/dimensions, thus in the mean one element is flipped.
struct BitflipMutator {

	/// \brief Default c'tor.
	BitflipMutator() : m_mutationStrength(0) {}

	/// \brief Initializes the operator for the supplied fitness function.
	///
	/// \param [in] f Instance of the objective function to initialize the operator for.
	template<typename Function>
	void init(const Function &f) {
		m_mutationStrength = 1./f.numberOfVariables();
	}

	/// \brief Mutates the supplied individual.
	/// 
	/// \param [in,out] ind Individual to be mutated.
	template<typename IndividualType>
	void operator()(IndividualType &ind) {

		for (unsigned int i = 0; i < ind.searchPoint().size(); i++) {
			if (Rng::coinToss(m_mutationStrength)) {
				ind.searchPoint()[ i ] = !ind.searchPoint()[ i ];
			}
		}
	}

	/// \brief Serializes this instance to the supplied archive.
	/// 
	/// \param [in,out] archive The archive to serialize to.
	/// \param [in] version Version information (optional and not used here).
	template<typename Archive>
	void serialize(Archive &archive, const unsigned int version) {
		archive &m_mutationStrength;
	}

	double m_mutationStrength;
};
}

#endif
