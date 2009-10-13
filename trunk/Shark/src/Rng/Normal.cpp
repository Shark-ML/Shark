//===========================================================================
/*!
 *  \file Normal.cpp
 *
 *  \brief Implements methods for class Normal that simulates a
 *         "normal", i.e. gaussian distribution.
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


#include <SharkDefs.h>
#include <Rng/Normal.h>



//========================================================================
/*!
 *  \brief Creates a new instance of the normal random number
 *         generator, initializes the range of the numbers and
 *         internal variables.
 *
 *  The mean value #pMean and the standard deviation #pStdDev of
 *  the distribution are initialized, where
 *  \f$pStdDev = \sqrt{\mbox{variance}}\f$. <br>
 *  Internal variables are also initialized to
 *  allow proper working. <br>
 *  For this instance, the default pseudo random number generator
 *  as member of class RandomVar is used.
 *
 *  \param mean     the mean value of the gaussian distribution,
 *                  the default is "0" for the standard distribution
 *  \param variance the variance of the gaussian distribution,
 *                  the default is "1" for the standard distribution
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
Normal::Normal(double mean, double variance)
		: pMean(mean),
		pStdDev(sqrt(variance)),
		iset(false),
		gset(0)      // to avoid uninitialized memory read warnings
{
}


//========================================================================
/*!
 *  \brief Creates a new normal random generator instance by
 *         using the pseudo random number generator "r" for the determination
 *         of random values and initializes the range of the numbers and
 *         internal variables.
 *
 *  Each instance of a normal random number generator is based
 *  on a generator, that is defined in class RNG and returns uniformally
 *  pseudo random numbers of the interval (0,1).
 *  By default, this is a global generator named RNG::globalRng and
 *  included as member in class RandomVar. <br>
 *  Here another pseudo random number generator \em r is used instead. <br>
 *  Additionally to defining the used pseudo random number generator,
 *  the mean value and the standard deviation of the distribution and internal
 *  variables are initialized.
 *
 *  \param mean     the mean value #pMean of the gaussian distribution
 *  \param variance the variance of the gaussian distribution from which
 *                  the standard deviation #pStdDev as
 *                  \f$pStdDev = \sqrt{\mbox{variance}}\f$ is calculated
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
Normal::Normal(double mean, double variance, RNG& r)
		: RandomVar< double >(r),
		pMean(mean),
		pStdDev(sqrt(variance)),
		iset(false),
		gset(0)      // to avoid uninitialized memory read warnings
{
}




//========================================================================
/*!
 *  \brief Returns a normally distributed random number for the
 *         distribution with mean value and standard deviation given in #pMean
 *         and #pStdDev.
 *
 *  This method performs the \em inverse \em transformation of the
 *  original uniformally distributed random numbers of the interval
 *  (0,1) created by the used pseudo random number generator to
 *  the type of the normal distribution. <br>
 *  For the transformation of the random number the Box-M&uuml;ller
 *  method is used as described in the class description. <br>
 *  Besides the random number returned at once, a second random number
 *  is stored in an internal variable. The next time calling this
 *  method, this stored number is returned.
 *
 *  \return the generated normally distributed random number
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
double Normal::operator()()
{
//
//	We actually generate two IID normal distribution variables.
//	We cache the one & return the other.
//

	double fac, rsq, v, w;

	if (! iset) {
		do {
			v   = 2 * rng() - 1;
			w   = 2 * rng() - 1;
			rsq = v * v + w * w;
		}
		while (rsq >= 1 || rsq == 0);

		fac  = sqrt(-2 * log(rsq) / rsq);
		gset = v * fac;
		iset = true;

		return pMean + w * fac * pStdDev;
	}
	else {
		iset = false;
		return pMean + gset * pStdDev;
	}
}


//========================================================================
/*!
 *  \brief Returns a normally distributed random number for the
 *         distribution with mean value "mean" and variance "variance".
 *
 *  For the generation of the random number the Box-M&uuml;ller
 *  method is used as described in the class description. <br>
 *  Besides the random number returned at once, a second random number
 *  is stored in an internal variable. The next time calling this
 *  method, this stored number is returned.
 *
 *  \return the generated normally distributed random number
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
double Normal::operator()(double mean, double variance)
{
	return operator()() *(sqrt(variance) / pStdDev)
		   + (mean - pMean);
}


//========================================================================
/*!
 *  \brief Returns the probability for the occurrence of "x"
 *         for a normal distribution with given mean value
 *         #pMean and standard deviation #pStdDev.
 *
 *  \param  x the value for which the occurrence probability will be
 *            returned
 *  \return the occurrence probability for \em x
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
double Normal::p(const double& x) const
{
	double y = (x - pMean) / pStdDev;

	return exp(-y*y / 2) / (Sqrt2PI * pStdDev);
}

