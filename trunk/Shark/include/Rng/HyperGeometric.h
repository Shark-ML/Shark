//===========================================================================
/*!
 *  \file HyperGeometric.h
 *
 *  \brief Contains a class that simulates a "Hyper %Geometric distribution".
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


#ifndef __HYPERGEOMETRIC_H
#define __HYPERGEOMETRIC_H

#include "Rng/RandomVar.h"

//===========================================================================
/*!
 *  \brief This class simulates a "Hyper %Geometric distribution".
 *
 *  This class is derived from class RandomVar and the uniformally
 *  distributed pseudo random number values of the interval (0,1)
 *  are transformed to type "double" of the Hyper %Geometric distribution. <br>
 *  The Hyper %Geometric distribution is used, when you have two types
 *  of events. Guess, the number of events of type \f$T_1\f$ is \f$N_1\f$
 *  and the number of events of type \f$T_2\f$ is \f$N_2\f$, so we have
 *  a total number \f$N = N_1 + N_2\f$ of events. Guess also, that we
 *  define the probability \f$p\f$ as the relation of the number of
 *  events of type \f$T_1\f$ to the total number of events
 *  (\f$p = \frac{N_1}{N}\f$). Then the distribution is given by
 *
 *  \f$
 *      P(k) = {pN \choose k} {N(1 - p) \choose n - k} / {N \choose k}
 *  \f$
 *
 *  with \f$p \cdot N\f$ and \f$N \cdot (1 - p)\f$ being always integer
 *  values. \f$P(k)\f$ calculates
 *  the probability, that you have \f$k\f$ events of type \f$T_1\f$
 *  in a sample of size \f$n \leq N\f$. <br>
 *  In contrast to the Binomial distribution, the Hyper %Geometric
 *  distribution takes into account, that probability \f$p\f$
 *  changes after each of the \f$n\f$ trials. <br>
 *  Taking the "urn model", the Hyper %Geometric distribution simulates
 *  the drawing of balls from the urn \em without returning the drawn balls
 *  to the urn.
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
class HyperGeometric : public RandomVar< double >
{
public:



	//! Creates a new Hyper %Geometric random generator instance and
	//! initializes the mean value and the variance of the
	//! distribution.
	HyperGeometric(double mean = 0, double variance = 1);

	//! Creates a new Hyper %Geometric random generator instance by
	//! using the pseudo random number generator "r" for the determination
	//! of random values and initializes the mean value and the variance
	//! of the distribution.
	HyperGeometric(double mean, double variance, RNG& r);

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

	//! Sets the current mean value of the distribution as stored
	//! in #pMean to the new value "newMean".
	void   mean(double newMean);

	//! Sets the current variance of the distribution as stored
	//! in #pVariance to the new value "newVar".
	void   variance(double newVar);

	//! Returns a Hyper %Geometric random number for the distribution
	//! with a probability \f$p = \frac{N_1}{N}\f$ that is calculated from
	//! the preset mean value #pMean and variance #pVariance.
	double operator()();

	//! Just a dummy method to allow instantiation of the class.
	double p(const double&) const;


protected:

	//! The distribution's mean value \f$\mu\f$, that is used to calculate
	//! the probability \f$p = \frac{N_1}{N}\f$ by
	//! \f$p = \frac{1}{2} \cdot (1 - \sqrt{\frac{\sigma^2 / \mu^2 - 1}{\sigma^2 / \mu^2 + 1}})\f$,
	//! where \f$\sigma^2\f$ is the distribution's variance as stored in
	//! #pVariance.
	double pMean;


	//! The distribution's variance \f$\sigma^2\f$, that is used to calculate
	//! probability \f$p = \frac{N_1}{N}\f$ by
	//! \f$p = \frac{1}{2} \cdot (1 - \sqrt{\frac{\sigma^2 / \mu^2 - 1}{\sigma^2 / \mu^2 + 1}})\f$,
	//! where \f$\mu\f$ is the distribution's mean value as stored in
	//! #pMean.
	double pVariance;

private:

	// The probability "p", i.e. the relation of the number of
	// events of type T1 to the total number of events given
	// by "pP = N1/N".
	double pP;

	// For a description see "HyperGeometric.cpp".
	void   setState();
};

#endif  /* !__HYPERGEOMETRIC_H */

