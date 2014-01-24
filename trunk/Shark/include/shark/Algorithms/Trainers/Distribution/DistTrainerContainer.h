/*!
 * 
 * \file        DistTrainerContainer.h
 *
 * \brief       Container for known distribution trainers.
 * 
 * 
 *
 * \author      B. Li
 * \date        2012
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
#ifndef SHARK_ALGORITHMS_TRAINERS_DISTRIBUTION_I_DIST_TRAINER_CONTAINER_H
#define SHARK_ALGORITHMS_TRAINERS_DISTRIBUTION_I_DIST_TRAINER_CONTAINER_H

#include "shark/Algorithms/Trainers/Distribution/NormalTrainer.h"

namespace shark {

/// Container for known distribution trainers
class DistTrainerContainer
{
public:

	/// Getter/setter for normal distribution
	/// @{
	const NormalTrainer& getNormalTrainer() const { return m_normalTrainer; }
	void setNormalTrainer(const NormalTrainer& normalTrainer) { m_normalTrainer = normalTrainer; }
	/// @}

	// Other distributions go here

private:

	NormalTrainer m_normalTrainer;
};

} // namespace shark {

#endif // SHARK_ALGORITHMS_TRAINERS_DISTRIBUTION_I_DIST_TRAINER_CONTAINER_H
