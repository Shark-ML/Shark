//===========================================================================
/*!
 * 
 *
 * \brief       Abstract class for statistical distributions
 * 
 * 
 * 
 *
 * \author      B. Li
 * \date        2012
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
