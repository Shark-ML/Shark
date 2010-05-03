//===========================================================================
/*!
 *  \file LogNormal.h
 *
 *  \brief Contains a class that simulates a "Log %Normal distribution".
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


#ifndef __LOGNORMAL_H
#define __LOGNORMAL_H


#include <cmath>
#include <SharkDefs.h>
#include <Rng/Normal.h>


//===========================================================================
/*!
 *  \brief This class simulates a "Log %Normal distribution".
 *
 *  This class is derived from class Normal and the random values
 *  returned by it are of type "double", simulating the Log %Normal
 *  distribution given by the equation: <br>
 *
 *  \f$
 *      f_{log}(x) =
 *          \frac{1}{\sqrt{2 \pi \sigma^2 x}} \cdot exp
 *          \left(- \frac{(ln x - \mu)^2}{2 \sigma^2}\right)
 *  \f$
 *
 *  where \f$x > 0\f$ and the log normal distribution results from the
 *  normal (gaussian) distribution \f$f_n(x)\f$ by
 *  \f$f_{log}(x) = \frac{1}{x}f_n(ln x)\f$. <br>
 *  \f$\mu\f$ is the mean value or peak of the distribution
 *  curve, stored in #logMean and \f$\sigma\f$ is the distribution's
 *  standard deviation, with the variance \f$V(x) = \sigma^2\f$ stored
 *  in #logVariance. <br>
 *  A special form of calculation is used to save time.
 *  For details please refer to
 *  "Simulation, Modelling & Analysis" by Law & Kelton, pp. 260. <br>
 *  Below you can see the Log #Normal distribution for the standard
 *  values
 *  \f$\mbox{mean} = \sqrt{e}\f$ and \f$\mbox{standard deviation} = e(e-1)\f$:
 *
 *  \image html logNormal.png
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 *
 */
class LogNormal : private Normal
{
public:

	//! Creates a new instance of the Log %Normal random number
	//! generator, initializes the range of the numbers and
	//! internal variables.
	LogNormal(double mean = SqrtE, double variance = M_E * (M_E - 1));


	//! Creates a new log normal random generator instance by
	//! using the pseudo random number generator "r" for the determination
	//! of random values and initializes the range of the numbers and
	//! internal variables.
	LogNormal(double mean, double variance, RNG& r);

	//========================================================================
	/*!
	 *  \brief Returns the current mean value #logMean of the random
	 *         numbers distribution.
	 *
	 *  \return the mean value \f$\mu\f$ as stored in #logMean
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
	double mean() const
	{
		return logMean;
	}


	//========================================================================
	/*!
	 *  \brief Returns the current variance #logVariance of the random
	 *         numbers distribution.
	 *
	 *  \return the variance \f$V(X)=\sigma^2\f$ as stored in #logVariance
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
	double variance() const
	{
		return logVariance;
	}


	//! Sets the current mean value #logMean of the distribution
	//! to the new value "newMean" and prepares internal variables
	//! for the future random number generation.
	void   mean(double newMean);


	//! Sets the current variance #logVariance of the distribution
	//! to the new value "newVar" and prepares internal variables
	//! for the future random number generation.
	void   variance(double newVar);

	//! Returns a log normally distributed random number for a
	//! distribution with mean value #logMean and variance
	//! #logVariance.
	double operator()();


	//! Returns the probability for the occurrence of "x"
	//! for a log normal distribution with given mean value
	//! #logMean and variance #logVariance.
	double p(const double&) const;


	//! Returns a log normally distributed random number for a
	//! distribution with mean value "mean" and variance
	//! "variance".
	double operator()(double mean, double variance);

protected:

	//! The mean value \f$\mu\f$ of the Log Normal distribution.
	double logMean;

	//! The variance \f$V(X)=\sigma^2\f$ of the distribution,
	//! where \f$\sigma\f$ is the distribution's standard deviation.
	double logVariance;

private:

	// See "LogNormal.cpp".
	void   setState();
};

#endif  /* !__LOGNORMAL_H */





