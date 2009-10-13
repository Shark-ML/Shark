//===========================================================================
/*!
 *  \file Erlang.h
 *
 *  \brief Contains a class that simulates an %Erlang distribution.
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


#ifndef __ERLANG_H
#define __ERLANG_H

#include "Rng/RandomVar.h"


//===========================================================================
/*!
 *  \brief This class simulates an Erlang distribution.
 *
 *  This class is derived from class RandomVar and the uniformally
 *  distributed pseudo random number values of the interval (0,1)
 *  are transformed to type double of the %Erlang distribution. <br>
 *  The %Erlang distribution is more practical than the (Negative) Exponential
 *  distribution, taking into account, that many processes need a
 *  minimum time. For this, the %Erlang distribution has an additional
 *  parameter \f$k = 1,2,3,...\f$, known as "the order" of the distribution.
 *  For every \f$k\f$ there is an %Erlang distribution, with the orders
 *  \f$k=1,2,3\f$ being relevant in practice. <br>
 *  The greater \f$k\f$ the more improbable are short times and the
 *  less the standard deviation is. <br>
 *  For \f$k = 1\f$ the %Erlang distribution is identical to the
 *  (Negative) Exponential distribution (see NegExponential). <br>
 *  The %Erlang distribution is given by:
 *
 *  \f$
 *      f(x) = (k \cdot \alpha)^k \cdot x^{(k - 1)} \cdot e^{-k \alpha x}
 *  \f$
 *
 *  for the scale \f$\alpha > 0\f$ and the order \f$k > 0\f$. <br>
 *  Below you can see the distribution for the parameter \f$\alpha = 0.5\f$
 *  for the distributions with the orders \f$k = 1, 2, 3\f$:
 *
 *  \image html erlang.png
 *  \image latex erlang.eps
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
class Erlang : public RandomVar< double >
{
public:

	//! Creates a new %Erlang random generator instance and
	//! initializes the mean value and the variance of the
	//! distribution.
	Erlang(double mean = 0, double variance = 1);


	//! Creates a new %Erlang random generator instance by
	//! using the pseudo random number generator \em r for the determination
	//! of random values and initializes the mean value and the variance
	//! of the distribution.
	Erlang(double mean, double variance, RNG& r);


//========================================================================
	/*!
	 *  \brief Returns the current mean value of the distribution as stored
	 *         in #pMean.
	 *
	 *  \return the current mean value \f$\mu\f$ (#pMean)
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
	 *  \brief Returns the current variance of the distribution as stored
	 *         in #pVariance.
	 *
	 *  \return the current variance \f$\sigma^2\f$ (#pVariance)
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
		return pVariance;
	}


//========================================================================
	/*!
	 *  \brief Sets the current mean value of the distribution as stored
	 *         in #pMean to the new value \em newMean.
	 *
	 *  The distribution's mean value \f$\mu\f$ is set to \em newMean and then
	 *  the distribution's order \f$k\f$ and scale factor \f$\alpha\f$ are
	 *  recalculated, because this calculation is based on the mean value. <br>
	 *  The distribution's order \f$k\f$ is defined by
	 *  \f$k = \frac{\mu^2}{\sigma^2+0.5}\f$,
	 *  where \f$\sigma^2\f$ is the distribution's variance as stored in
	 *  #pVariance. The mean value is also used to calculate the
	 *  scale factor \f$\alpha\f$ by \f$\alpha = \frac{k}{\mu}\f$. <br>
	 *  \f$k\f$ will always be an integer value \f$\geq 1\f$ and
	 *  \f$\alpha\f$ will always be a real number \f$> 0\f$.
	 *
	 *  \param newMean the new value for the distribution's mean value
	 *                 \f$\mu\f$ (#pMean). If \em newMean is less than
	 *                 or equal to zero then the distribution's scale
	 *                 factor \f$\alpha\f$ is set to 0.5.
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
		pMean     = newMean; setState();
	}


//========================================================================
	/*!
	 *  \brief Sets the current variance of the distribution as stored
	 *         in #pVariance to the new value \em newVar.
	 *
	 *  The distribution's mean value \f$\sigma^2\f$ is set to \em newVar and then
	 *  the distribution's order \f$k\f$ and scale factor \f$\alpha\f$ are
	 *  recalculated, because this calculation is based on the variance. <br>
	 *  The distribution's order \f$k\f$ is defined by
	 *  \f$k = \frac{\mu^2}{\sigma^2+0.5}\f$,
	 *  where \f$\mu\f$ is the distribution's mean value as stored in
	 *  #pMean. This order is then used to calculate the
	 *  scale factor \f$\alpha\f$ by \f$\alpha = \frac{k}{\mu}\f$. <br>
	 *  \f$k\f$ will always be an integer value \f$\geq 1\f$ and
	 *  \f$\alpha\f$ will always be a real number \f$> 0\f$.
	 *
	 *  \param newVar the new value for the distribution's variance
	 *                \f$\sigma^2\f$ (#pVariance)
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
		pVariance = newVar;  setState();
	}


	//! Returns an %Erlang random number for the distribution
	//! with order \em k and scale factor \em a.
	double operator()(unsigned k, double a);

	//! Returns an %Erlang random number for the distribution
	//! with order and scale factor that are calculated from
	//! the preset mean value #pMean and variance #pVariance.
	double operator()();

	//! Returns the probability for the occurrence of random number
	//! \em x for the %Erlang distribution with order and scale factor
	//! calculated from the preset mean value #pMean and variance
	//! #pVariance.
	double p(const double&) const;

protected:

	//! The distribution's mean value \f$\mu\f$, that is used to calculate
	//! the distribution's order \f$k\f$ by
	//! \f$k = \frac{\mu^2}{\sigma^2+0.5}\f$,
	//! where \f$\sigma^2\f$ is the distribution's variance as stored in
	//! #pVariance. The mean value is also used to calculate the
	//! scale factor \f$\alpha\f$ by \f$\alpha = \frac{k}{\mu}\f$.
	double pMean;

	//! The distribution's variance value \f$\sigma^2\f$, that is used to
	//! calculate the distribution's order \f$k\f$ by
	//! \f$k = \frac{\mu^2}{\sigma^2+0.5}\f$,
	//! where \f$\mu\f$ is the distribution's mean value as stored in
	//! #pMean. Because the scale factor \f$\alpha\f$ is calculated
	//! by \f$\alpha = \frac{k}{\mu}\f$, the variance is also
	//! used for the calculation of this factor.
	double pVariance;

private:

	// The distribution's parameter \em k (the order of the distribution)
	unsigned pK;

	// The distributions parameter \f$\alpha\f$ (the scale of the distribution)
	double   pA;

	// See Erlang.cpp for description.
	void setState();
};

#endif  /* !__ERLANG_H */

