//===========================================================================
/*!
 *  \file Poisson.h
 *
 *  \brief Contains a class that simulates a "%Poisson distribution".
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

#ifndef __POISSON_H
#define __POISSON_H

#include "Rng/RandomVar.h"


//===========================================================================
/*!
 *  \brief This class simulates a "%Poisson distribution".
 *
 *  This class is derived from class RandomVar and the uniformally
 *  distributed pseudo random number values of the interval (0,1)
 *  are transformed to type "long", indicating the number
 *  of "hits" out of a given number of trials with a given
 *  probability for a single hit. <br>
 *  This is similar to the Binomial distribution, but the %Poisson
 *  distribution delivers an approximation, when the number of trials
 *  \f$n\f$ is very large and the probability \f$p\f$ for a single hit
 *  is very small. <br>
 *  Then the %Poisson distribution is given for all \f$x \geq 0\f$ as:
 *
 *  \f$
 *      p_x^{(n)} = P(X = x) = ( \lambda^x \cdot e^{- \lambda} ) / x!
 *  \f$
 *
 *  where \f$\lambda > 0\f$ is the parameter of the distribution and
 *  given by \f$\lambda = n \cdot p\f$. <br>
 *  Below you can see the distribution for the parameter values
 *  \f$\lambda = 2.0\f$ and \f$\lambda = 5.0\f$:
 *
 *  \image html poisson.png
 *  \image latex poisson.eps
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
class Poisson : public RandomVar< long >
{
public:

	//! Creates a new instance of the poisson random number
	//! generator and initializes the distribution's parameter value.
	Poisson(double mean = 0);

	//! Creates a new poisson random generator instance by
	//! using the pseudo random number generator "r" for the determination
	//! of random values and initializes the distribution's parameter value.
	Poisson(double mean, RNG& r);


//========================================================================
	/*!
	 *  \brief Returns the distribution parameter's current value.
	 *
	 *  The current value of #pMean that represents the value of \f$\lambda\f$ in
	 *  the poisson distribution (see class description) is returned.
	 *
	 *  \return the current value of #pMean (parameter \f$\lambda\f$)
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
	 *  \brief Sets the distribution parameter's current value to "newMean".
	 *
	 *  The current value of #pMean that represents the value of \f$\lambda\f$ in
	 *  the poisson distribution (see class description) is set to \em newMean.
	 *
	 *  \param newMean the new value of #pMean (parameter \f$\lambda\f$)
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


	//! Returns a poisson random number, i.e. the number of hits
	//! for parameter \f$\lambda = \mbox{mean}\f$.
	long operator()(double mean);

	//! Returns a poisson random number, i.e. the number of hits
	//! for parameter \f$\lambda\f$ as stored in #pMean.
	long operator()();

	//! Returns the probability for "x" hits when the distribution's
	//! parameter \f$\lambda\f$ is set to #pMean.
	double p(const long&) const;

protected:

	//! The value of \f$\lambda\f$ in the poisson distribution
	//! (see class description).
	double pMean;
};

#endif  /* !__POISSON_H */






