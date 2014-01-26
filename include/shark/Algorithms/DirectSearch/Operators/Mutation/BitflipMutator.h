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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_MUTATION_BITFLIP_MUTATION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_MUTATION_BITFLIP_MUTATION_H

#include <shark/Rng/GlobalRng.h>

namespace shark {

/**
 * \brief Bitflip mutation operator.
 */
struct BitflipMutator {

	/**
	 * \brief Default c'tor.
	 */
	BitflipMutator(double mutationStrength) : m_mutationStrength(mutationStrength) {}

	/**
	 * \brief Initializes the operator for the supplied fitness function.
	 * \tparam Function Objective function type. Needs to be model of AbstractVectorSpaceObjectiveFunction.
	 * \param [in] f Instance of the objective function to initialize the operator for.
	 */
	template<typename Function>
	void init(const Function &f) {
		m_mutationStrength = 1./f.numberOfVariables();
	}

	/**
	 * \brief Mutates the supplied individual.
	 * \tparam IndividualType Type of the individual, needs to provider operator*
	 *  for accessing the actual search point.
	 * \param [in,out] ind Individual to be mutated.
	 */
	template<typename IndividualType>
	void operator()(IndividualType &ind) {

		for (unsigned int i = 0; i < (*ind).size(); i++) {
			if (Rng::coinToss(m_mutationStrength)) {
				(*ind)[ i ] = !(*ind)[ i ];
			}
		}
	}

	/**
	 * \brief Serializes this instance to the supplied archive.
	 * \tparam Archive The type of the archive the instance shall be serialized to.
	 * \param [in,out] archive The archive to serialize to.
	 * \param [in] version Version information (optional and not used here).
	 */
	template<typename Archive>
	void serialize(Archive &archive, const unsigned int version) {
		archive &m_mutationStrength;
	}

	double m_mutationStrength;
};
}

#endif
