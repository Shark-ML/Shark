//===========================================================================
/*!
 *  \file Normal.h
 *
 *  \brief Contains a class that simulates a "normal", i.e.
 *         gaussian distribution.
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
 *  <BR>
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


#ifndef __NORMAL_H
#define __NORMAL_H

#include <cmath>
#include "Rng/RandomVar.h"


//===========================================================================
/*!
 *  \brief This class simulates a "normal" distribution.
 *
 *  This class is derived from class RandomVar and the uniformally
 *  distributed pseudo random number values of the interval (0,1)
 *  are transformed to type "double", simulating the normal
 *  or gaussian distribution with the equation: <br>
 *
 *  \f$
 *      f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot exp \left(- \frac{(x- \mu)^2}{2 \sigma^2}\right)
 *  \f$
 *
 *  where \f$\mu\f$ is the mean value or peak of the distribution
 *  curve, stored in #pMean and \f$\sigma\f$ is the distribution's
 *  standard deviation, stored in #pStdDev. <br>
 *  Because the used "real" random generator produces uniformally distributed
 *  numbers, the Box-M&uuml;ller method is used to create two 2-dimensional
 *  standard normally distributed random numbers from two 2-dimensional
 *  uniformally distributed random numbers between \f$0\f$ and \f$1\f$. <br>
 *  One of the generated normally distributed numbers is then returned
 *  and the other one is stored for the next time. <br>
 *  For details about the Box-M&uuml;ller method please refer to
 *  "Simulation, Modelling & Analysis" by Law & Kelton, pp. 259. <br>
 *  Below you can see the distribution for the standard normal distribution
 *  with \f$\mu = 0\f$ and \f$\sigma = 1\f$:
 *
 *  \image html normal.png
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
class Normal : public RandomVar< double >
{
public:

	//! Creates a new instance of the normal random number
	//! generator, initializes the range of the numbers and
	//! internal variables.
	Normal(double mean = 0, double variance = 1);


	//! Creates a new normal random generator instance by
	//! using the pseudo random number generator "r" for the determination
	//! of random values and initializes the range of the numbers and
	//! internal variables.
	Normal(double mean, double variance, RNG& r);

	//========================================================================
	/*!
	 *  \brief Initializes the pseudo random number generator used by this 
	 *         class with value "s".
	 *
	 *  The pseudo random number generator as defined in class RNG is
	 *  initialized by using the seed value \em s. <br>
	 *  Additionally, an internal variable is initialized to allow
	 *  proper working.
	 *
	 *  \param s initialization value for the pseudo random number generator
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
	 *  \sa RandomVar::seed
	 *
	 */
	void seed(long s)
	{
		RandomVar< double >::seed(s); iset = false;
	}


	//========================================================================
	/*!
	 *  \brief Returns the current mean value #pMean of the random
	 *         numbers distribution.
	 *
	 *  \return the mean value \f$\mu\f$ as stored in #pMean
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
		return pMean;
	}


	//========================================================================
	/*!
	 *  \brief Returns the current variance of the random
	 *         numbers distribution.
	 *
	 *  The variance \f$V(X)\f$ is calculated from the standard 
	 *  deviation \f$\sigma\f$ as stored in #pStdDev by
	 *
	 *  \f$
	 *      V(X) = \sigma^2
	 *  \f$
	 *
	 *  \return the variance \f$V(X)\f$
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
		return pStdDev * pStdDev;
	}


	//========================================================================
	/*!
	 *  \brief Sets the current mean value #pMean of the distribution
	 *         to the new value "newMean".
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
	void   mean(double newMean)
	{
		pMean = newMean;
	}

	//========================================================================
	/*!
	 *  \brief Sets the current variance of the distribution
	 *         to the new value "newVar".
	 *
	 *  The variance \f$V(X)\f$ is set to the new value
	 *  "newVar" and from this the new standard deviation
	 *  \f$\sigma\f$ is calculated as \f$\sigma = \sqrt{V(X)}\f$
	 *  and stored in #pStdDev.
	 *
	 *  \param  newVar the new variance of the distribution
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
	void   variance(double newVar)
	{
		pStdDev = sqrt(newVar);
	}

	//! Returns a normally distributed random number for the
	//! distribution with mean value "mean" and variance "variance".
	double operator()(double mean, double variance);

	//! Returns a normally distributed random number for the
	//! distribution with mean value and standard deviation given in #pMean
	//! and #pStdDev.
	double operator()();

	//! Returns the probability for the occurrence of "x"
	//! for a normal distribution with given mean value
	//! #pMean and standard deviation #pStdDev.
	double p(const double&) const;

protected:

	//! The mean value \f$\mu\f$ of the normal distribution.
	double pMean;

	//! The standard deviation \f$\sigma\f$ of the distribution.
	double pStdDev;

private:

	// If set to "true", the second normally distributed
	// random number, generated one step before, was
	// stored in "gset" and can be returned at once. Otherwise,
	// two new random numbers must be generated.
	bool   iset;

	// The second generated normally distributed random number
	// is stored here, when "iset" is set to "true".
	double gset;

};

#endif  /* !__NORMAL_H */

