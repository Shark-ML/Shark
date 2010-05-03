//===========================================================================
/*!
 *  \file Poisson.cpp
 *
 *  \brief Implements methods for class Poisson that simulates a
 *         "%Poisson distribution".
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
#include "Rng/Poisson.h"
#include "Rng/Uniform.h"
#include "Rng/Normal.h"


//========================================================================
/*!
 *  \brief Creates a new instance of the poisson random number
 *         generator and initializes the distribution's parameter value.
 *
 *  The mean value #pMean that represents the value of \f$\lambda\f$ in
 *  the poisson distribution (see class description) is initialized
 *  by \em mean. <br>
 *  For this instance, the default pseudo random number generator
 *  as member of class RandomVar is used.
 *
 *  \param mean the initial value for \f$\lambda\f$ stored in #pMean, the
 *              default is "0"
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
Poisson::Poisson(double mean)
		: pMean(mean)
{}


//========================================================================
/*!
 *  \brief Creates a new poisson random generator instance by
 *         using the pseudo random number generator "r" for the determination
 *         of random values and initializes the distribution's parameter value.
 *
 *  Each instance of a poisson random number generator is based
 *  on a generator, that is defined in class RNG and returns uniformally
 *  pseudo random numbers of the interval (0,1).
 *  By default, this is a global generator named RNG::globalRng and
 *  included as member in class RandomVar. <br>
 *  Here another pseudo random number generator \em r is used instead. <br>
 *  Additionally to defining the used pseudo random number generator,
 *  the parameter \f$\lambda\f$ (see class description) as stored in
 *  #pMean is initialized.
 *
 *  \param mean the initial value for \f$\lambda\f$ stored in #pMean
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
Poisson::Poisson(double mean, RNG& r)
		: RandomVar< long >(r), pMean(mean)
{}


//========================================================================
/*!
 *  \brief Returns a poisson random number, i.e. the number of hits
 *         for parameter \f$\lambda = \mbox{mean}\f$.
 *
 *  This method performs the \em inverse \em transformation of the
 *  original uniformally distributed random numbers of the interval
 *  (0,1) created by the used pseudo random number generator to
 *  the type of the poisson distribution. <br>
 *  The implementation of this method is based on "Numerical Recipes
 *  in C", p. 221.
 *
 *  \param mean the value for \f$\lambda\f$
 *  \return the number of hits (events)
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes
 *      mt, 2002-02-01: <br>
 *      A positive integer drawn from a poisson distribution with given
 *      "mean"; is case mean > 100, a (positive) gauss number
 *      ( floor(mean+gauss(sqrt(mean))+.5) ) is returned.
 *
 *  \par Status
 *      stable
 *
 */
long Poisson::operator()(double mean)
{
	Normal gauss;
	Uniform uni;
	long count = 0;
	double bound, product;
	int i;

	if (mean > 100) {
		i = (int) floor(mean + gauss(sqrt(mean), 1.) + .5);
		return (i > 0) ? (long) i : 0;
	}
	if (mean >= 0) {
		bound   = exp(-mean);
		product = uni();

		while (product >= bound) {
			count++;
			product *= uni();
		}
	}

	return count;
}


//========================================================================
/*!
 *  \brief Returns a poisson random number, i.e. the number of hits
 *         for parameter \f$\lambda\f$ as stored in #pMean.
 *
 *  \return the number of hits (events)
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
long Poisson::operator()()
{
	return operator()(pMean);
}


//
// Calculates the factorial (j!).
//
static double fac(long j)
{
	double f = 1;

	while (j > 0)
		f *= (double) j--;

	return f;
}


//========================================================================
/*!
 *  \brief Returns the probability for "x" hits when the distribution's
 *         parameter \f$\lambda\f$ is set to #pMean.
 *
 *  \param x the number of hits
 *  \return the probability for \em x hits or "0" if \f$x < 0\f$
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
double Poisson::p(const long& x) const
{
	return x >= 0 ? pow(pMean, x) * exp(-pMean) / fac(x) : 0;
}

