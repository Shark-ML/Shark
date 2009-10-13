//===========================================================================
/*!
 *  \file HyperGeometric.cpp
 *
 *  \brief Implements methods for class HyperGeometric, that simulates
 *         a "Hyper %Geometric distribution".
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
#include "Rng/HyperGeometric.h"


//========================================================================
/*!
 *  \brief Creates a new Hyper %Geometric random generator instance and
 *         initializes the mean value and the variance of the
 *         distribution.
 *
 *  The distribution's mean value \f$\mu\f$, that is stored in #pMean
 *  is initialized by \em mean and the distribution's variance
 *  \f$\sigma^2\f$, that is stored in #pVariance is initialized
 *  by \em variance. <br>
 *  Then the mean value and the variance are used to calculate
 *  the probability \f$p = \frac{N_1}{N}\f$ by
 *  \f$p = \frac{1}{2} \cdot (1 - \sqrt{\frac{\sigma^2 / \mu^2 - 1}{\sigma^2 / \mu^2 + 1}})\f$. <br>
 *  \f$p\f$ will always be a value \f$\geq 0\f$. <br>
 *  If you are calling this constructor with the default values
 *  \em mean = 0 and \em variance = 1, then the probability \f$p\f$
 *  will be the default value \f$p = 0.\f$. <br>
 *  For this instance, the default pseudo random number generator
 *  as member of class RandomVar is used.
 *
 *  \param mean     the initial value for the distribution's mean value
 *                  \f$\mu\f$ from which the probability \f$p\f$
 *                  is calculated. The default
 *                  mean value is zero. If you use a mean value
 *                  less than or equal to zero, the probability
 *                  \f$p\f$ will be set to "0."
 *  \param variance the initial value for the distribution's variance
 *                  \f$\sigma^2\f$ from which the probability \f$p\f$
 *                  is calculated. The default variance is "1"
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
HyperGeometric::HyperGeometric(double mean, double variance)
		: pMean(mean), pVariance(variance)
{
	setState();
}


//========================================================================
/*!
 *  \brief Creates a new Hyper %Geometric random generator instance by
 *         using the pseudo random number generator "r" for the determination
 *         of random values and initializes the mean value and the variance
 *         of the distribution.
 *
 *  Each instance of a Hyper %Geometric random number generator is based
 *  on a generator, that is defined in class RNG and returns uniformally
 *  pseudo random numbers of the interval (0,1).
 *  By default, this is a global generator named RNG::globalRng and
 *  included as member in class RandomVar. <br>
 *  Here another pseudo random number generator \em r is used instead. <br>
 *  Additionally to defining the used pseudo random number generator,
 *  the distribution's mean value, that is stored in #pMean
 *  is initialized by \em mean and the distribution's variance
 *  \f$\sigma^2\f$, that is stored in #pVariance is initialized
 *  by \em variance. <br>
 *  Then the mean value and the variance are used to calculate
 *  the probability \f$p = \frac{N_1}{N}\f$ by
 *  \f$p = \frac{1}{2} \cdot (1 - \sqrt{\frac{\sigma^2 / \mu^2 - 1}{\sigma^2 / \mu^2 + 1}})\f$. <br>
 *  \f$p\f$ will always be a value \f$\geq 0\f$.
 *
 *  \param mean     the initial value for the distribution's mean value
 *                  \f$\mu\f$ from which the probability \f$p\f$
 *                  is calculated. If you use a mean value
 *                  less than or equal to zero, the probability
 *                  \f$p\f$ will be set to "0."
 *  \param variance the initial value for the distribution's variance
 *                  \f$\sigma^2\f$ from which the probability \f$p\f$
 *                  is calculated.
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
HyperGeometric::HyperGeometric(double mean, double variance, RNG& r)
		: RandomVar< double >(r), pMean(mean), pVariance(variance)
{
	setState();
}


// Calculates the probability "p", i.e. the relation of the number of
// events of type T1 to the total number of events given
// by "p = N1/N" from the stored mean value and the
// variance. "p" will always be a real number greater than or equal
// to zero. If the stored mean value is zero, then "p" will be set
// to "0.", as for the default values "mean = 0" and "variance = 1".
// Change: 2002-04-16, ra:
// If the mean value is equal to zero or the calculation
// of "p" results in a negative value then "p" will be set to "0." now.
//
void HyperGeometric::setState()
{
	if (pMean == 0) {
		pP = 0.;
		return;
	}

	double z = pVariance / (pMean * pMean);
	pP = (1 - sqrt((z - 1) / (z + 1))) / 2;

	if (pP < 0.) pP = 0.;
	return;
}



//========================================================================
/*!
 *  \brief Sets the current mean value of the distribution as stored
 *         in #pMean to the new value "newMean".
 *
 *  The distribution's mean value \f$\mu\f$ is set to \em newMean and then
 *  the probability \f$p = \frac{N_1}{N}\f$ is recalculated by
 *  \f$p = \frac{1}{2} \cdot (1 - \sqrt{\frac{\sigma^2 / \mu^2 - 1}{\sigma^2 / \mu^2 + 1}})\f$,
 *  where \f$\sigma^2\f$ is the distribution's variance as stored in
 *  #pVariance.
 *
 *  \param newMean the new value for the distribution's mean value
 *                 \f$\mu\f$ (#pMean)
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
void HyperGeometric::mean(double newMean)
{
	pMean = newMean;
	setState();
}


//========================================================================
/*!
 *  \brief Sets the current variance of the distribution as stored
 *         in #pVariance to the new value "newVar".
 *
 *  The distribution's variance \f$\sigma^2\f$ is set to \em newVar and then
 *  the probability \f$p = \frac{N_1}{N}\f$ is recalculated by
 *  \f$p = \frac{1}{2} \cdot (1 - \sqrt{\frac{\sigma^2 / \mu^2 - 1}{\sigma^2 / \mu^2 + 1}})\f$,
 *  where \f$\mu\f$ is the distribution's mean value as stored in
 *  #pMean.
 *
 *  \param newVariance the new value for the distribution's variance
 *                 \f$\sigma^2\f$ (#pVariance)
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
void HyperGeometric::variance(double newVariance)
{
	pVariance = newVariance;
	setState();
}



//========================================================================
/*!
 *  \brief Returns a Hyper %Geometric random number for the distribution
 *         with a probability \f$p = \frac{N_1}{N}\f$ that is calculated from
 *         the preset mean value #pMean and variance #pVariance.
 *
 *  The mean value \f$\mu\f$as stored in #pMean and the variance
 *  \f$\sigma^2\f$as stored in #pVariance are used to calculate
 *  the probability \f$p = \frac{N_1}{N}\f$ by
 *  \f$p = \frac{1}{2} \cdot (1 - \sqrt{\frac{\sigma^2 / \mu^2 - 1}{\sigma^2 / \mu^2 + 1}})\f$. <br>
 *  \f$p\f$ will always be a value \f$\geq 0\f$. <br>
 *  If you are calling this operator with a preset mean value
 *  \f$\mu \leq 0\f$, then the probability \f$p\f$
 *  will be the default value \f$p = 0.\f$.
 *
 *  \return a random number for the Hyper %Geometric distribution with a
 *          probability \f$p\f$ based on #pMean and #pVariance
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes
 *          none
 *
 *  \par Status
 *      stable
 *
 */
double HyperGeometric::operator()()
{
	return -pMean * log(rng()) / (2 *(rng() > pP ? 1 - pP : pP));
}



//========================================================================
/*!
 *  \brief Just a dummy method to allow instantiation of the class.
 *
 *  This method, defined as purely virtual in class RandomVar
 *  has to be implemented to allow the instantiation of class
 *  #HyperGeometric.
 *
 *  \param  x has no function here
 *  \return always "0."
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes
 *           none
 *
 *  \par Status
 *      stable
 *
 */
double HyperGeometric::p(const double& x) const
{
	return 0.; // !!!
}












