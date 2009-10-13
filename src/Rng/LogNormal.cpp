//===========================================================================
/*!
 *  \file LogNormal.cpp
 *
 *  \brief Implements methods for class LogNormal that simulates a
 *         "Log %Normal distribution".
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
#include <Rng/LogNormal.h>


//========================================================================
/*!
 *  \brief Creates a new instance of the Log %Normal random number
 *         generator, initializes the range of the numbers and
 *         internal variables.
 *
 *  The mean value #logMean and the variance #logVariance of
 *  the distribution are initialized. <br>
 *  Internal variables are also initialized to
 *  speed up the generation of future random numbers. <br>
 *  For this instance, the default pseudo random number generator
 *  as member of class RandomVar is used.
 *
 *  \param mean     the mean value of the log normal distribution,
 *                  the default is \f$\sqrt{e}\f$
 *  \param variance the variance of the log normal distribution,
 *                  the default is \f$e(e-1)\f$
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
LogNormal::LogNormal(double mean, double variance)
		: logMean(mean),
		logVariance(variance)
{
	setState();
}


//========================================================================
/*!
 *  \brief Creates a new log normal random generator instance by
 *         using the pseudo random number generator "r" for the determination
 *         of random values and initializes the range of the numbers and
 *         internal variables.
 *
 *  Each instance of a log normal random number generator is based
 *  on a generator, that is defined in class RNG and returns uniformally
 *  pseudo random numbers of the interval (0,1).
 *  By default, this is a global generator named RNG::globalRng and
 *  included as member in class RandomVar. <br>
 *  Here another pseudo random number generator \em r is used instead. <br>
 *  Additionally to defining the used pseudo random number generator,
 *  the mean value and the variance of the distribution and internal
 *  variables are initialized.
 *
 *  \param mean     the mean value #logMean of the log normal distribution
 *  \param variance the variance #logVariance of the log normal distribution
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
LogNormal::LogNormal(double mean, double variance, RNG& r)
		: Normal(0, 1, r),
		logMean(mean),
		logVariance(variance)
{
	setState();
}


// Given a mean value and a variance for the log normal distribution
// this method will prepare the mean value and the standard deviation
// of the inherited normal distribution in a way, that the generation
// of log normally distributed random numbers will take only a single
// simple step (see operator()()).
//
void LogNormal::setState()
{
	double m2 = logMean * logMean;
	pMean     = log(m2 / sqrt(m2 + logVariance));
	pStdDev   = sqrt(log((m2 + logVariance) / m2));
}


//========================================================================
/*!
 *  \brief Sets the current mean value #logMean of the distribution
 *         to the new value "newMean" and prepares internal variables
 *         for the future random number generation.
 *
 *  \param  newMean the new mean value for the distribution
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
void LogNormal::mean(double newMean)
{
	logMean = newMean;
	setState();
}


//========================================================================
/*!
 *  \brief Sets the current variance #logVariance of the distribution
 *         to the new value "newVar" and prepares internal variables
 *         for the future random number generation.
 *
 *  The variance \f$V(X)=\sigma^2\f$ is set to the new value
 *  "newVar".
 *
 *  \param  newVariance the new variance of the distribution
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
void LogNormal::variance(double newVariance)
{
	logVariance = newVariance;
	setState();
}



//========================================================================
/*!
 *  \brief Returns a log normally distributed random number for a
 *         distribution with mean value #logMean and variance
 *         #logVariance.
 *
 *  To create a random number a special method (see class description)
 *  based on the Box-M&uuml;ller method is used.
 *
 *  \return a log normally distributed random number
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
 *  \sa class Normal
 *
 */
double LogNormal::operator()()
{
	return exp(Normal::operator()());
}


//========================================================================
/*!
 *  \brief Returns a log normally distributed random number for a
 *         distribution with mean value "mean" and variance
 *         "variance".
 *
 *  This method performs the \em inverse \em transformation of the
 *  original uniformally distributed random numbers of the interval
 *  (0,1) created by the used pseudo random number generator to
 *  the type of the log normal distribution. <br>
 *  To create such a transformed random number, a special
 *  method (see class description)
 *  based on the Box-M&uuml;ller method is used.
 *
 *  \return a log normally distributed random number
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
 *  \sa class Normal
 *
 */
double LogNormal::operator()(double mean, double variance)
{
	double m2 = mean * mean;
	double normMean = log(m2 / sqrt(m2 + variance));
	double normVariance = sqrt(log((m2 + variance) / m2));

	return exp(Normal::operator()(normMean, normVariance));
}


//========================================================================
/*!
 *  \brief Returns the probability for the occurrence of "x"
 *         for a log normal distribution with given mean value
 *         #logMean and variance #logVariance.
 *
 *  \param  x the value \f$x > 0\f$ for which the occurrence probability
 *            will be returned
 *  \return the occurrence probability for \em x or "0" if \f$x \leq 0\f$
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
double LogNormal::p(const double& x) const
{
	double y = (log(x) - pMean) / pStdDev;

	return x > 0 ? exp(-y*y / 2) / (Sqrt2PI * pStdDev * x) : 0;
}

