//===========================================================================
/*!
 *  \file Binomial.h
 *
 *  \brief Contains a class that simulates a "Binomial distribution".
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

#ifndef __BINOMIAL_H
#define __BINOMIAL_H

#include "Rng/RandomVar.h"


//===========================================================================
/*!
 *  \brief This class simulates a "Binomial distribution".
 *
 *  This class is derived from class RandomVar and the uniformally
 *  distributed pseudo random number values of the interval (0,1)
 *  are transformed to type "long", indicating the number
 *  of "hits" out of a given number of trials with a given
 *  probability for a single hit. <br>
 *  The probability for a special number \f$x\f$ of hits can also be
 *  calculated by using the %Bernoulli formula given as
 *
 *  \f$p_x^{(n)} = P(X = x) = {n \choose x} \cdot p^x \cdot q^{n-x}\f$
 *
 *  where \f$p\f$ is the probability for a single hit, \f$q = 1 - p\f$ and
 *  \f$n\f$ is the total number of trials. <br>
 *  Taking the "urn model", the %Binomial distribution simulates
 *  the drawing of \f$n\f$ balls from the urn \em with returning the drawn
 *  balls to the urn. <br>
 *  Below you can see the distribution for the parameter values
 *  \f$n = 8\f$ and \f$p = 0.3\f$ and \f$p = 0.5\f$, respectively:
 *
 *  \image html binomial.png
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
class Binomial : public RandomVar< long >
{
public:


	//! Creates a new instance of the binomial random number
	//! generator and initializes values for the #Bernoulli formula.
	Binomial(unsigned n = 1, double p = 0.5);

	//! Creates a new binomial random generator instance by
	//! using the pseudo random number generator "r" for the determination
	//! of random values and initializes values for the #Bernoulli formula.
	Binomial(unsigned n, double p, RNG& r);


//========================================================================
	/*!
	 *  \brief Returns the probability #pP for a hit.
	 *
	 *  \return the probability #pP for a hit
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
	double   pp() const
	{
		return pP;
	}


//========================================================================
	/*!
	 *  \brief Sets the probability #pP for a hit to the new value "newP".
	 *
	 *  \param newP the new value for the probability #pP
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
	void     pp(double newP)
	{
		pP = newP;
	}


//========================================================================
	/*!
	 *  \brief Returns the number of trials #pN.
	 *
	 *  \return the number of trials #pN
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
	unsigned n() const
	{
		return pN;
	}


//========================================================================
	/*!
	 *  \brief Sets the number of trials #pN to the new value "newN".
	 *
	 *  \param newN the new value for the number of trials #pN
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
	void     n(unsigned newN)
	{
		pN = newN;
	}

	//! Returns a binomial random number, i.e. the number of hits
	//! in "n" trials, with a probability of "p" for a single hit.
	long operator()(unsigned n, double p);

	//! Returns a binomial random number, i.e. the number of hits
	//! in #pN trials, with a probability of #pP for a single hit.
	long operator()();

	//! Just a dummy method that is needed for instantiation
	//! of class Binomial.
	double p(const long&) const;

	//! Returns the probability for "x" hits out of #pN trials,
	//! when the probability for a single hit is #pP.
	double p(const unsigned x) const;

	//! Returns the probability for "x" hits out of "n" trials,
	//! when the probability for a single hit is "p".
	double p(const unsigned n, const double p, const unsigned x) const;

	//! Returns the binomial coefficient "n choose x".
	unsigned binominalCoefficient(const unsigned n, const unsigned x) const;

	//! Returns the factorial "tt!".
	unsigned factorial(const unsigned tt) const;

protected:

	//! The number of trials, i.e. the value \f$n\f$ in the #Bernoulli
	//! formula
	//! \f$p_x^{(n)} = P(X = x) = {n \choose x} \cdot p^x \cdot q^{n-x}\f$.
	unsigned pN;

	//! The probability for a "hit", i.e. the value \f$p\f$ in the #Bernoulli
	//! formula
	//! \f$p_x^{(n)} = P(X = x) = {n \choose x} \cdot p^x \cdot q^{n-x}\f$.
	double   pP;
};

#endif  /* !__BINOMIAL_H */






