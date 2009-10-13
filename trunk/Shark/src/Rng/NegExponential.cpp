//===========================================================================
/*!
 *  \file NegExponential.cpp
 *
 *  \brief Implements methods for class NegExponential that simulates a
 *         "negative exponential distribution".
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


#include <cmath>
#include "Rng/NegExponential.h"


//========================================================================
/*!
 *  \brief Creates a new instance of the negative exponential random number
 *         generator and initializes the parameter \f$\lambda\f$.
 *
 *  The parameter \f$\lambda\f$ that is stored in #pMean is initialized
 *  by \em mean. <br>
 *  For this instance, the default pseudo random number generator
 *  as member of class RandomVar is used.
 *
 *  \param mean initial value for #pMean (parameter \f$\lambda\f$),
 *              the default is "0". Notice that \em mean must be
 *              greater than zero, otherwise the methods for
 *              returning random numbers or the probability
 *              for a random number will always return "0"
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
NegExponential::NegExponential(double mean)
		: pMean(mean)
{}


//========================================================================
/*!
 *  \brief Creates a new neg exponential random generator instance by
 *         using the pseudo random number generator "r" for the determination
 *         of random values and initializes the distribution's
 *         parameter \f$\lambda\f$.
 *
 *  Each instance of a neg exponential random number generator is based
 *  on a generator, that is defined in class RNG and returns uniformally
 *  pseudo random numbers of the interval (0,1).
 *  By default, this is a global generator named RNG::globalRng and
 *  included as member in class RandomVar. <br>
 *  Here another pseudo random number generator \em r is used instead. <br>
 *  Additionally to defining the used pseudo random number generator,
 *  #pMean (parameter \f$\lambda\f$) is initialized by \em mean. <br>
 *
 *  \param mean initial value for #pMean (parameter \f$\lambda\f$).
 *              Notice that \em mean must be
 *              greater than zero, otherwise the methods for
 *              returning random numbers or the probability
 *              for a random number will always return "0"
 *  \param r the pseudo random number generator that is used
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
NegExponential::NegExponential(double mean, RNG& r)
		: RandomVar< double >(r), pMean(mean)
{}


//========================================================================
/*!
 *  \brief Returns a negative exponential random number for the
 *         parameter \f$\lambda\f$ as preset in #pMean.
 *
 *  \return the negative exponential random number for
 *          \f$\lambda = \f$ #pMean or "0" if #pMean is
 *          less than or equal to zero
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes
 *      2002-04-16, ra: <br>
 *      Now the parameter \f$\lambda\f$ is checked for
 *      being greater than "0"
 *
 *  \par Status
 *      stable
 *
 */
double NegExponential::operator()()
{
	if (pMean > 0) return operator()(pMean);
	else return 0.;
}


//========================================================================
/*!
 *  \brief Returns the probability for the occurrence of
 *         random number "x".
 *
 *  The probability is returned for the parameter \f$\lambda\f$
 *  that is set to #pMean.
 *
 *  \param x the random number for which the probability is
 *           returned
 *  \return the probability for the occurrence of random number
 *          \em x or "0" if \f$x < 0\f$
 *
 *  \author  M. Kreutz
 *  \date    1998-08-17
 *
 *  \par Changes
 *      2002-04-16, ra: <br>
 *      This method was just a dummy method before. Now it really
 *      returns the probability.
 *
 *  \par Status
 *      stable
 *
 */
double NegExponential::p(const double& x) const
{
	return x >= 0 ? pMean * exp(- pMean * x) : 0.;
}




