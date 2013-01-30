/**
*
*  \brief Container for known distribution trainers.
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
