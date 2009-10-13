//===========================================================================
/*!
 *  \file Uniform.cpp
 *
 *  \brief Implements methods for class Uniform that simulates a
 *         "uniform distribution".
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Copyright (c) 1995,1998:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      Rng
 * 
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Rng. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
//===========================================================================

#include "Rng/Uniform.h"

//========================================================================
/*!
 *  \brief Creates a new instance of the uniform random number
 *         generator and initializes the lower and upper bound.
 *
 *  The lower bound #pLow and the upper bound #pHigh
 *  for the interval, from which the random numbers are
 *  taken, are initialized. <br>
 *  For this instance, the default pseudo random number generator
 *  as member of class RandomVar is used.
 *
 *  \param lo initial value for the lower bound #pLow, by default set to "0"
 *  \param hi initial value for the upper bound #pHigh, by default set to "1"
 *  \return none
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
Uniform::Uniform(double lo, double hi)
		: pLow(lo), pHigh(hi)
{}


//========================================================================
/*!
 *  \brief Creates a new uniform random generator instance by
 *         using the pseudo random number generator "r" for the determination
 *         of random values and initializes the interval bounds
 *         for the random numbers.
 *
 *  Each instance of a uniform random number generator is based
 *  on a "real" random number generator, that is defined in
 *  class RNG and returns random numbers in the interval (0,1).
 *  Normally this is a global generator named RNG::globalRng.
 *  The random values returned by this "real" random number
 *  generator are then transformed into the type of the
 *  uniform random number generator (double).<br>
 *  See classes RNG and RandomVar for details. <br>
 *  Additionally to defining the used "real" random number generator,
 *  the lower and the upper bound for the random numbers interval
 *  are initialized.
 *
 *  \param lo initial value for the lower bound #pLow
 *  \param hi initial value for the upper bound #pHigh
 *  \param r the pseudo random number generator that is used
 *  \return none
 *
 *  \author  M. Kreutz
 *  \date    1998-08-17
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
Uniform::Uniform(double lo, double hi, RNG& r)
		: RandomVar< double >(r), pLow(lo), pHigh(hi)
{}


//========================================================================
/*!
 *  \brief Returns a uniformally distributed random number
 *         from the interval [#pLow, #pHigh[.
 *
 *  A random number \f$rn\f$ with #pLow \f$\leq rn <\f$ #pHigh
 *  is returned.
 *
 *  \return the random number \f$rn\f$
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
double Uniform::operator()()
{
	return operator()(pLow, pHigh);
}



//========================================================================
/*!
 *  \brief Returns the probability for the occurrence of random number
 *         "x".
 *
 *  The probability \f$p = \frac{1}{\mbox{interval length}}\f$
 *  is returned, where the \em interval \em length is given
 *  by #pHigh - #pLow. If \em x is not a member of the interval,
 *  "0" is returned instead.
 *
 *  \return the probability \f$p\f$ or "0" if \em x is not an interval
 *          member
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
double Uniform::p(const double& x) const
{
	return x >= pLow && x < pHigh ? 1 / (pHigh - pLow) : 0;
}








