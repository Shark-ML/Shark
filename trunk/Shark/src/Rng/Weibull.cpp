//===========================================================================
/*!
 *  \file Weibull.cpp
 *
 *  \brief Implements methods for class Weibull, that simulates
 *         a "%Weibull distribution".
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


#include "Rng/Weibull.h"
#include "math.h"
#include "SharkDefs.h"
#include <iostream>


//========================================================================
/*!
 *  \brief Creates a new instance of the %Weibull random number
 *         generator and initializes the distribution's parameters.
 *
 *  The distribution's parameter \f$\alpha\f$ as stored in #pAlpha
 *  and \f$\beta\f$ as stored in #pBeta are initialized. <br>
 *  For this instance, the default pseudo random number generator
 *  as member of class RandomVar is used.
 *
 *  \param alpha  the parameter \f$\alpha\f$ of the distribution, the default
 *                value is "1"
 *  \param beta   the parameter \f$\beta\f$ of the distribution, the default
 *                value is "1"
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
Weibull::Weibull(double alpha, double beta)
		: pAlpha(alpha), pBeta(beta)
{}


//========================================================================
/*!
 *  \brief Creates a new %Weibull random generator instance by
 *         using the pseudo random number generator "r" for the determination
 *         of random values and initializes the distribution's parameters.
 *
 *  Each instance of a %Weibull random number generator is based
 *  on a generator, that is defined in class RNG and returns uniformally
 *  pseudo random numbers of the interval (0,1).
 *  By default, this is a global generator named RNG::globalRng and
 *  included as member in class RandomVar. <br>
 *  Here another pseudo random number generator \em r is used instead. <br>
 *  Additionally to defining the used pseudo random number generator,
 *  the distribution's parameters \f$alpha\f$ as stored in #pAlpha
 *  and \f$\beta\f$ as stored in #pBeta are initialized.
 *
 *  \param alpha  the parameter \f$\alpha\f$ of the distribution
 *  \param beta   the parameter \f$\beta\f$ of the distribution
 *  \param r    the pseudo random number generator that is used
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
Weibull::Weibull(double alpha, double beta, RNG& r)
		: RandomVar< double >(r), pAlpha(alpha), pBeta(beta)
{}


//========================================================================
/*!
 *  \brief For the current distribution parameters \f$\alpha\f$ and
 *         \f$\beta\f$, this method returns a %Weibull random number.
 *
 *  For the current distribution's parameters \f$\alpha\f$ as stored in
 *  #pAlpha and \f$\beta\f$ as stored in #pBeta a %Weibull random number
 *  is returned.
 *
 *  \return a %Weibull random number
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
double Weibull::operator()()
{
	return operator()(pAlpha, pBeta);
}



//========================================================================
/*!
 *  \brief Returns the probability for the occurrence of random
 *         number "x" for the %Weibull distribution with the
 *         values "a" for parameter \f$\alpha\f$ and "b" for
 *         parameter \f$\beta\f$.
 *
 *  \param  a the value for the distribution's parameter \f$\alpha\f$
 *  \param  b the value for the distribution's parameter \f$\beta\f$
 *  \param  x the random number for which the occurrence probability
 *            will be returned. If \f$x \leq 0\f$ then the method will
 *            exit with an error message
 *  \return the occurrence probability for random number \em x
 *
 *  \throw SahrkException
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
double Weibull::p(const double &a, const double &b, const double &x) const
{
	if (x <= 0) {
	  throw SHARKEXCEPTION("Weibull distribution not defined for x <= 0");
	}
	return a / b *(-pow(x / b, a)) * pow(x / b, a - 1.);
}


//========================================================================
/*!
 *  \brief Returns the probability for the occurrence of random
 *         number "x" for the %Weibull distribution with the
 *         parameter values \f$\alpha\f$ as stored in #pAlpha and
 *         \f$\beta\f$ as stored in #pBeta.
 *
 *  \param  x the random number for which the occurrence probability
 *            will be returned. If \f$x \leq 0\f$ then the method will
 *            exit with an error message
 *  \return the occurrence probability for random number \em x
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
double Weibull::p(const double &x) const
{
	return Weibull::p(pAlpha, pBeta, x);
}


