//===========================================================================
/*!
 * 
 *
 * \brief       AbstractEvolutionStrategy
 * 
 * 
 *
 * \author
 * \date        2018
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
//===========================================================================
#ifndef SHARK_ABSTRACTEVOLUTIONSTRATEGY_H
#define SHARK_ABSTRACTEVOLUTIONSTRATEGY_H

namespace shark {

/// \brief An optimizer that applies an evolution stragtegy
/// 
/// This interface serves the purpose of allowing an evolution strategy based
/// optimizer to supply a population set to an external evaluation mechanism
/// as opposed to only expose the `step` function. This puts more control
/// in the user of the optimizer as evaluation is shifted to be fullly controlled
/// by the user of the optimizer instead of having the optimizer handle evaluation
/// internally.
///
/// Also when init() is called as offered by the AbstractOptimizer interface, the function
/// is required to have the CAN_PROPOSE_STARTING_POINT flag.
template<typename IndividualType>
class AbstractEvolutionStrategy {
public:
    ///\brief Returns a population from the current distribution
    virtual std::vector<IndividualType> generateOffspring() const = 0;

    ///\brief Update the internal distribution
    virtual void updatePopulation(std::vector<IndividualType> const& offspring) = 0;
};

}

#endif // SHARK_ABSTRACTEVOLUTIONSTRATEGY_H
