//===========================================================================
/*!
 *  \brief Abstract class for statistical distributions
 *
 *
 *  \author  B. Li
 *  \date    2012
 *
 *  \par Copyright (c) 2012:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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
 *
 */
//===========================================================================
#ifndef SHARK_RNG_ABSTRACT_DISTRIBUTION_H
#define SHARK_RNG_ABSTRACT_DISTRIBUTION_H

#include "shark/Core/Exception.h"
#include "shark/Core/Math.h"

namespace shark {

/// Abstract class for distributions
class AbstractDistribution
{
public:
	/// Dtor
	virtual ~AbstractDistribution() {}

	/// Calculate probability for a given input
	/// @param x the input for calculating probability
	/// @return probability of input
	virtual double p(double x) const = 0;

	/// Calculate log(p(x))
	///
	/// std::log can get -inf before it returns NaN. shark::safeLog tries to save the day, however is not perfect.
	/// The only real solution is to implement a function logP inside the distributions which returns the energy of the state
	/// @note subclasses should implement their own version of this function instead of replying on the default
	/// implementation unless you are pretty sure what you are doing.
	///
	/// @param x the input for calculating log of probability
	/// @return log of probability of input
	virtual double logP(double x) const { return safeLog(p(x)); }
};

} // namespace shark {

#endif // SHARK_RNG_ABSTRACT_DISTRIBUTION_H
