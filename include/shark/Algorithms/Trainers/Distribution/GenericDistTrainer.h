/**
*
*  \brief Implementations of various distribution trainers.
*
*  \author  B. Li
*  \date    2012
*
*  \par Copyright (c) 2012
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
*/
#ifndef SHARK_ALGORITHMS_TRAINERS_DISTRIBUTION_GENERIC_DIST_TRAINER_H
#define SHARK_ALGORITHMS_TRAINERS_DISTRIBUTION_GENERIC_DIST_TRAINER_H

#include "shark/Algorithms/Trainers/Distribution/DistTrainerContainer.h"
#include "shark/Algorithms/Trainers/Distribution/NormalTrainer.h"
#include "shark/Rng/Normal.h"
#include "shark/Rng/Rng.h"
#include "shark/Rng/Uniform.h"

namespace shark {

/// The trainer which is smart enough to train different kinds of distributions
///
/// @note all train functions should be reentrant
class GenericDistTrainer
:
	public DistTrainerContainer
{
public:

	/// Train an abstract distribution
	/// @param abstractDist the distribution we want to train
	/// @param input the input data used for training the dist
	/// @throw throw shark exception if training attempt for this distribution failed
	void train(AbstractDistribution& abstractDist, const std::vector<double>& input) const
	{
		// We have to do manual dispatching here unless distributions are trainer-aware/-friendly

		if (tryTrain<Normal<DefaultRngType> >(abstractDist, getNormalTrainer(), input))
			return;
		if (tryTrain<Normal<FastRngType> >(abstractDist, getNormalTrainer(), input))
			return;

		// Other distributions go here

		throw SHARKEXCEPTION("No trainer for this distribution.");
	}

private:

	/// Try to train an abstract distribution with given concrete distribution type
	/// @param abstractDist the abstract distribution
	/// @param trainer the trainer to be used for training the distribution
	/// @param input the input data
	/// @tparam DistType the type of concrete distribution
	/// @tparam TrainerType the type of trainer
	/// @return true if the training attempt succeeded, false otherwise
	template <typename DistType, typename TrainerType>
	bool tryTrain(AbstractDistribution& abstractDist, const TrainerType& trainer, const std::vector<double>& input) const
	{
		DistType* dist = dynamic_cast<DistType*>(&abstractDist);
		if (dist)
		{
			trainer.train(*dist, input);
			return true;
		}
		else
		{
			return false;
		}
	}
};

} // namespace shark {

#endif // SHARK_ALGORITHMS_TRAINERS_DISTRIBUTION_GENERIC_DIST_TRAINER_H
