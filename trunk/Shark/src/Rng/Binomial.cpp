//===========================================================================
/*!
 *  \file Binomial.cpp
 *
 *  \brief Implements methods for class Binomial that simulates a
 *         "Binomial distribution".
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

#include "Rng/Binomial.h"


//========================================================================
/*!
 *  \brief Creates a new instance of the binomial random number
 *         generator and initializes values for the #Bernoulli formula.
 *
 *  The number of trials #pN and the probability for a hit #pP
 *  for the #Bernoulli formula used by the random number generator
 *  are initialized. <br>
 *  For this instance, the default pseudo random number generator
 *  as member of class RandomVar is used.
 *
 *  \param n the number of trials #pN, set to "1" by default
 *  \param p the probability #pP for a hit, set to "0.5" by default
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
Binomial::Binomial(unsigned n, double p)
		: pN(n), pP(p)
{}


//========================================================================
/*!
 *  \brief Creates a new binomial random generator instance by
 *         using the pseudo random number generator "r" for the determination
 *         of random values and initializes values for the #Bernoulli formula.
 *
 *  Each instance of a binomial random number generator is based
 *  on a generator, that is defined in class RNG and returns uniformally
 *  pseudo random numbers of the interval (0,1).
 *  By default, this is a global generator named RNG::globalRng and
 *  included as member in class RandomVar. <br>
 *  Here another pseudo random number generator \em r is used instead. <br>
 *  Additionally to defining the used pseudo random number generator,
 *  the number of trials #pN and the probability for a hit #pP
 *  for the #Bernoulli formula used by the random number generator
 *  are initialized.
 *
 *  \param n   the number of trials #pN
 *  \param p   the probability #pP for a hit
 *  \param r the pseudo random number generator that is used
 *  \return none
 *
 *  \author  M. Kreutz
 *  \date    1998-08-17
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
Binomial::Binomial(unsigned n, double p, RNG& r)
		: RandomVar< long >(r), pN(n), pP(p)
{}


//========================================================================
/*!
 *  \brief Returns a binomial random number, i.e. the number of hits
 *         in "n" trials, with a probability of "p" for a single hit.
 *
 *  This method performs the \em inverse \em transformation of the
 *  original uniformally distributed random numbers of the interval
 *  (0,1) created by the used pseudo random number generator to
 *  the type of the binomial distribution.
 *
 *  \param n the number of trials
 *  \param p the probability for a single hit
 *  \return the number of hits
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
long Binomial::operator()(unsigned n, double p)
{
	long s = 0;

	while (n--)
		s += long(rng() + p);

	return s;
}


//========================================================================
/*!
 *  \brief Returns a binomial random number, i.e. the number of hits
 *         in #pN trials, with a probability of #pP for a single hit.
 *
 *  \return the number of hits
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
long Binomial::operator()()
{
	return operator()(pN, pP);
}


//========================================================================
/*!
 *  \brief Just a dummy method that is needed for instantiation
 *         of class Binomial.
 *
 *  To instantiate this class, this dummy method that is declared as purely
 *  virtual in class RandomVar, has to be implemented.
 *
 *  \param x not used
 *  \return always "0"
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
double Binomial::p(const long& x) const
{
	return 0;
}



//========================================================================
/*!
 *  \brief Returns the probability for "x" hits out of #pN trials,
 *         when the probability for a single hit is #pP.
 *
 *  The #Bernoulli formula, that calculates the probability for
 *  \em x hits out of \em n trials is here defined as
 *
 *  \f$
 *      p_x^{(n)} = P(X = x) = {n \choose x} \cdot p^x \cdot q^{n-x}
 *  \f$
 *
 *  Here, for \f$n\f$ and \f$p\f$, the values of #pN and #pP are used.
 *
 *  \param x the number of hits, i.e. the value of
 *           \f$x\f$ in the #Bernoulli formula
 *  \return the probability \f$p_x^{(n)}\f$
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
double Binomial::p(const unsigned x) const
{
	return Binomial::p(pN, pP, x);
}


//========================================================================
/*!
 *  \brief Returns the probability for "x" hits out of "n" trials,
 *         when the probability for a single hit is "p".
 *
 *  The #Bernoulli formula, that calculates the probability for
 *  \em x hits out of \em n trials is here defined as
 *
 *  \f$
 *      p_x^{(n)} = P(X = x) = {n \choose x} \cdot p^x \cdot q^{n-x}
 *  \f$
 *
 *  \param n the number of trials, i.e. the value of \f$n\f$
 *           in the #Bernoulli formula
 *  \param p the probability for a single hit, i.e. the value
 *           of \f$p\f$ (\f$q = 1 -p\f$) in the #Bernoulli formula
 *  \param x the number of hits, i.e. the value of
 *           \f$x\f$ in the #Bernoulli formula
 *  \return the probability \f$p_x^{(n)}\f$
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
double Binomial::p(const unsigned n, const double p, const unsigned x) const
{
	unsigned ii;
	double prob;
	double q = 1 - p; // faster for large n

	prob = static_cast<double>(Binomial::binominalCoefficient(n , x));

	// p^x:
	for (ii = 0; ii < x; ii++) {
		prob *= p;
	}

	// q^(n-x):
	for (ii = x; ii < n; ii++) {
		prob *= q;
	}

	return prob;
}


//========================================================================
/*!
 *  \brief Returns the binomial coefficient "n choose x".
 *
 *  The binomial coefficient is here defined as
 *
 *  \f$
 *      {n \choose x} = \frac{n \cdot (n-1) \cdot (n-2) \cdot \dots \cdot
 *                      (n-x+1)}{1 \cdot 2 \cdot \dots \cdot x} =
 *                      \prod_{i=1}^x \frac{n+1-i}{i}
 *  \f$
 *
 *  \param n the value of \f$n\f$ in \f${n \choose x}\f$
 *  \param x the value of \f$x\f$ in \f${n \choose x}\f$
 *  \return the binomial coefficient \f${n \choose x}\f$
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
unsigned Binomial::binominalCoefficient(const unsigned n, const unsigned x) const
{
	unsigned result;
	result = factorial(n);
	result /= (factorial(x) * factorial(n - x));
	return result;
}


//========================================================================
/*!
 *  \brief Returns the factorial "tt!".
 *
 *  The factorial of \em tt defined as
 *
 *  \f$
 *      tt! = tt \cdot (tt - 1) \cdot (tt - 2) \cdot \dots \cdot 2 \cdot 1
 *          = \prod_{i=1}^{tt} i
 *  \f$
 *
 *  \param tt the value of which the factorial is calculated
 *  \return \f$tt!\f$
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
unsigned Binomial::factorial(const unsigned tt) const
{
	unsigned result = 1;
	for (unsigned ii = 2; ii <= tt; ii++) {
		result *= ii;
	}
	return result;
}




