//===========================================================================
/*!
 *  \file Erlang.cpp
 *
 *  \brief Implements methods for a class Erlang, that simulates an
 *         %Erlang distribution.
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
#include "Rng/Erlang.h"


//========================================================================
/*!
 *  \brief Creates a new %Erlang random generator instance and
 *         initializes the mean value and the variance of the
 *         distribution.
 *
 *  The distribution's mean value \f$\mu\f$, that is stored in #pMean
 *  is initialized by \em mean and the distribution's variance
 *  \f$\sigma^2\f$, that is stored in #pVariance is initialized
 *  by \em variance. <br>
 *  Then the mean value and the variance are used to calculate
 *  the distribution's order \f$k\f$ by \f$k = \frac{\mu^2}{\sigma^2+0.5}\f$
 *  and the distribution's scale factor \f$\alpha\f$ by
 *  \f$\alpha = \frac{k}{\mu}\f$. <br>
 *  \f$k\f$ will always be an integer value \f$\geq 1\f$ and
 *  \f$\alpha\f$ will always be a real number \f$> 0\f$. <br>
 *  If you are calling this constructor with the default values
 *  \em mean = 0 and \em variance = 1, then the distribution's
 *  order will be the default value \f$k = 1\f$ and the scale
 *  factor will be the default value \f$\alpha = 0.5\f$. <br>
 *  For this instance, the default pseudo random number generator
 *  as member of class RandomVar is used.
 *
 *  \param mean     the initial value for the distribution's mean value
 *                  \f$\mu\f$ from which the distribution's parameters
 *                  \f$k\f$ and \f$\alpha\f$ are calculated. The default
 *                  mean value is zero. If you use a mean value
 *                  less than or equal to zero, the distribution's
 *                  scale factor \f$\alpha\f$ will be set to "0.5"
 *  \param variance the initial value for the distribution's variance
 *                  \f$\sigma^2\f$ from which the distribution's parameters
 *                  \f$k\f$ and \f$\alpha\f$ are calculated. The default
 *                  variance is "1"
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
Erlang::Erlang(double mean, double variance)
		: pMean(mean), pVariance(variance)
{
	setState();
}


//========================================================================
/*!
 *  \brief Creates a new %Erlang random generator instance by
 *         using the pseudo random number generator "r" for the determination
 *         of random values and initializes the mean value and the variance
 *         of the distribution.
 *
 *  Each instance of an %Erlang random number generator is based
 *  on a generator, that is defined in class RNG and returns uniformally
 *  pseudo random numbers of the interval (0,1).
 *  By default, this is a global generator named RNG::globalRng and
 *  included as member in class RandomVar. <br>
 *  Here another pseudo random number generator \em r is used instead. <br>
 *  Additionally to defining the used pseudo random number generator,
 *  the distribution's mean value \f$\mu\f$, that is stored in #pMean
 *  is initialized by \em mean and the distribution's variance
 *  \f$\sigma^2\f$, that is stored in #pVariance is initialized
 *  by \em variance. <br>
 *  Then the mean value and the variance are used to calculate
 *  the distribution's order \f$k\f$ by \f$k = \frac{\mu^2}{\sigma^2+0.5}\f$
 *  and the distribution's scale factor \f$\alpha\f$ by
 *  \f$\alpha = \frac{k}{\mu}\f$. <br>
 *  \f$k\f$ will always be an integer value \f$\geq 1\f$ and
 *  \f$\alpha\f$ will always be a real number \f$> 0\f$.
 *
 *  \param mean     the initial value for the distribution's mean value
 *                  \f$\mu\f$ from which the distribution's parameters
 *                  \f$k\f$ and \f$\alpha\f$ are calculated. If you use
 *                  a mean value
 *                  less than or equal to zero, the distribution's
 *                  scale factor \f$\alpha\f$ will be set to "0.5"
 *  \param variance the initial value for the distribution's variance
 *                  \f$\sigma^2\f$ from which the distribution's parameters
 *                  \f$k\f$ and \f$\alpha\f$ are calculated
 *  \param r        the pseudo random number generator that is used
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
Erlang::Erlang(double mean, double variance, RNG& r)
		: RandomVar< double >(r), pMean(mean), pVariance(variance)
{
	setState();
}


// Calculates the distribution's order "k" and the distribution's
// scale factor "alpha" from the stored mean value and the
// variance. "k" will always be an integer value greater than or equal
// to "1" and "alpha" will always be a double value greater than
// zero. For the default values "mean = 0" and "variance = 1"
// you will get "k = 1" and "alpha = 0.5".
// Change: 2002-04-16, ra:
// If the mean value is less than or equal to zero, the scale
// factor "alpha" will be set to "0.5" now.
//
void Erlang::setState()
{
	pK = unsigned(pMean * pMean / pVariance + 0.5);
	if (pK == 0) pK = 1;
	if (pMean > 0) pA = pK / pMean;
	else pA = 0.5;
}



//========================================================================
/*!
 *  \brief Returns an %Erlang random number for the distribution
 *         with order "k" and scale factor "a".
 *
 *  This method performs the \em inverse \em transformation of the
 *  original uniformally distributed random numbers of the interval
 *  (0,1) created by the used pseudo random number generator to
 *  the type of the %Erlang distribution.
 *
 *  \param k the distribution's order. \em k must be greater
 *           than zero, otherwise "0." will be returned
 *  \param a the distribution's scale \f$\alpha\f$. \em a must be greater
 *           than zero, otherwise "0." will be returned
 *  \return a random number for the %Erlang distribution with order
 *          \em k and scale factor \em a
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes
 *      2002-04-16, ra: <br>
 *      Now checks the distribution's parameters "k" and "a"
 *      for validity (both must be greater than zero) and
 *      returns zero if at least one parameter is not valid.
 *
 *  \par Status
 *      stable
 *
 */
double Erlang::operator()(unsigned k, double a)
{
	double prod = 1;

	if (k == 0 || a <= 0) return 0.;

	while (k--)
		prod *= rng();

	return -log(prod) / a;
}


//========================================================================
/*!
 *  \brief Returns an %Erlang random number for the distribution
 *         with order and scale factor that are calculated from
 *         the preset mean value #pMean and variance #pVariance.
 *
 *  The mean value \f$\mu\f$as stored in #pMean and the variance
 *  \f$\sigma^2\f$as stored in #pVariance are used to calculate
 *  the distribution's order \f$k\f$ by \f$k = \frac{\mu^2}{\sigma^2+0.5}\f$
 *  and the distribution's scale factor \f$\alpha\f$ by
 *  \f$\alpha = \frac{k}{\mu}\f$. <br>
 *  \f$k\f$ will always be an integer value \f$\geq 1\f$ and
 *  \f$\alpha\f$ will always be a real number \f$> 0\f$. <br>
 *  If you are calling this operator with a preset mean value
 *  \f$\mu \leq 0\f$, then the distribution's
 *  order will be the default value \f$k = 1\f$ and the scale
 *  factor will be the default value \f$\alpha = 0.5\f$.
 *
 *  \return a random number for the %Erlang distribution with order
 *          and scale factor based on #pMean and #pVariance
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes
 *      2002-04-16, ra: <br>
 *      Now checks the distribution's parameters "k" and "a"
 *      for validity (both must be greater than zero) and
 *      returns zero if at least one parameter is not valid.
 *
 *  \par Status
 *      stable
 *
 */
double Erlang::operator()()
{
	return operator()(pK, pA);
}


//========================================================================
/*!
 *  \brief Returns the probability for the occurrence of random number
 *         "x" for the %Erlang distribution with order and scale factor
 *         calculated from the preset mean value #pMean and variance
 *         #pVariance.
 *
 *  The mean value \f$\mu\f$as stored in #pMean and the variance
 *  \f$\sigma^2\f$ as stored in #pVariance are used to calculate
 *  the distribution's order \f$k\f$ by \f$k = \frac{\mu^2}{\sigma^2+0.5}\f$
 *  and the distribution's scale factor \f$\alpha\f$ by
 *  \f$\alpha = \frac{k}{\mu}\f$. <br>
 *  \f$k\f$ will always be an integer value \f$\geq 1\f$ and
 *  \f$\alpha\f$ will always be a real number \f$> 0\f$. <br>
 *  If you are calling this method with a preset mean value
 *  \f$\mu \leq 0\f$, then the distribution's
 *  order will be the default value \f$k = 1\f$ and the scale
 *  factor will be the default value \f$\alpha = 0.5\f$. <br>
 *  For the %Erlang distribution with the calculated mean value
 *  and variance, the occurrence probability of random number \em x
 *  is then returned.
 *
 *  \return x the random number for which the occurrence probability
 *            is returned. If \em x is less than zero the probability
 *            "0." will be returned
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes
 *      2002-04-16, ra: <br>
 *      Was just a dummy method before. Now it really calculates
 *      the probability.
 *
 *  \par Status
 *      stable
 *
 */
double Erlang::p(const double& x) const
{
	if (pK == 0 || pA <= 0 || x < 0) return 0.;

	return pow((static_cast< double >(pK) * pA), static_cast< double >(pK)) * pow(x, static_cast< double >(pK - 1)) * exp(- static_cast< double >(pK) * pA * x);
}








