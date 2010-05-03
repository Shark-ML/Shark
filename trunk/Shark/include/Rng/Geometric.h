//===========================================================================
/*!
 *  \file Geometric.h
 *
 *  \brief Contains a class that simulates a "%Geometric distribution".
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


#ifndef __GEOMETRIC_H
#define __GEOMETRIC_H

#include "Rng/RandomVar.h"


//===========================================================================
/*!
 *  \brief This class simulates a "%Geometric distribution".
 *
 *  This class is derived from class RandomVar and the uniformally
 *  distributed pseudo random number values of the interval (0,1)
 *  are transformed to type "long", simulating the geometric
 *  distribution. <br>
 *  The geometric distribution is based on the Bernoulli distribution, i.e.
 *  if you have a probability of \f$p\f$ for the occurrence of an event
 *  and following from this a probability of \f$(1-p)\f$ for the
 *  non-occurrence of this event and you are performing \f$x\f$
 *  independantly %Bernoulli trials, the geometric distribution
 *  gives you the number \f$x\f$ of trials after which the first success
 *  (event) occurs after \f$x-1\f$ fails (non-occurrences
 *  of the event). The probabilities for the first success at trial no.
 *  \f$x\f$ (the distribution function) is given for all \f$x > 0\f$ by:
 *
 *  \f$
 *      f(x) = p \cdot (1-p)^{x-1}
 *  \f$
 *
 *  For \f$x = 0\f$ the result is always \f$f(x) = 0\f$. <br>
 *  Below you can see the distribution function for the probabilities
 *  \f$p = 0.8\f$, \f$p = 0.5\f$ and \f$p = 0.2\f$ for a single
 *  %Bernoulli trial:
 *
 *  \image html geom.png
 *  \image latex geom.eps
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
class Geometric : public RandomVar< long >
{
public:

	//! Creates a new instance of the geometric random number
	//! generator and sets the probability for a single
	//! %Bernoulli trial.
	Geometric(double mean = 0);

	//! Creates a new geometric random generator instance by
	//! using the pseudo random number generator "r" for the determination
	//! of random values and sets the probability for a single
	//! %Bernoulli trial.
	Geometric(double mean, RNG& r);

//========================================================================
	/*!
	 *  \brief Returns the current probability for a single %Bernoulli
	 *         trial as stored in #pMean.
	 *
	 *  \return the probability #pMean for a single %Bernoulli trial
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
	 *  \brief Sets the probability #pMean for a single %Bernoulli
	 *         trial to "newMean".
	 *
	 *  \param newMean the new probability for a single %Bernoulli trial
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


	//! For a given probability "mean" for a single %Benoulli trial,
	//! this method returns the number of trials after which the
	//! first success occurs.
	long operator()(double mean);

	//! For the current probability for a single %Benoulli trial stored
	//! in #pMean, this method returns the number of trials after which the
	//! first success occurs.
	long operator()();

	//! For a number of trials "x", this method returns the
	//! probability that the first success occurs after this
	//! number of trials.
	double p(const long&) const;

protected:

	//! The probability \f$p\f$ for a single %Bernoulli trial.
	double pMean;
};

#endif  /* !__GEOMETRIC_H */

