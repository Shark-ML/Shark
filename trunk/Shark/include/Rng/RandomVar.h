//===========================================================================
/*!
 *  \file RandomVar.h
 *
 *  \brief Contains a template class that will define the basic operations
 *         for all random number generators of library "#Rng", mostly
 *         for Monte Carlo simulations.
 *
 *  \author  M. Kreutz
 *  \date    1998-08-17
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

#ifndef __RANDOMVAR_H
#define __RANDOMVAR_H

#ifdef _WIN32
// disable warning C4804: suspicious bool comparison
// occurs during instantiation of vector< bool >
#pragma warning(disable: 4804)
#endif

#include <cmath>
#include <vector>
#include "Rng/RNG.h"


//===========================================================================
/*!
 *  \brief Template class used as base for all random number generators
 *         of library "#Rng".
 *
 *  Here the basic operations for all random number generators
 *  of library "Rng" are defined, especially for Monte Carlo simulations
 *  (see #setMonteCarlo). <br>
 *  For the generation of random numbers the generator for
 *  uniformally distributed pseudo random numbers as defined
 *  in class RNG is used as member of this class.
 *
 *  \author  M. Kreutz
 *  \date    1998-08-17
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 *
 *  \sa #setMonteCarlo
 *
 */
template < class T >
class RandomVar
{
public:

	virtual ~RandomVar()
	{ }

	//! Returns a new random number of the distribution defined by
	//! the function \f$F(X)\f$ in the derived class, i.e.
	//! normally the "inverse transformation" of the
	//! uniformally distributed pseudo random numbers of the
	//! interval (0,1) to the distribution of the derived class
	//! takes place here. So, this method is purely virtual and must
	//! be overloaded in the derived classes.
	virtual T operator()() = 0;


	//! Returns the probability of the occurrence of the random
	//! number value given as parameter of this method. <br>
	//! This method is purely virtual and must be overloaded
	//! in the derived classes, because the probability of the
	//! parameter depends on the type of distribution given by
	//! the type of random number generator used.
	virtual double p(const T&) const = 0;


	//========================================================================
	/*!
	 *  \brief Initializes the pseudo random number generator with the seed
	 *         value "s".
	 *
	 *  The pseudo random number generator as defined in class RNG is
	 *  initialized by using the value \em s.
	 *
	 *  \param s initialization (seed) value for the pseudo random number 
	 *           generator
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
	virtual void   seed(long s)
	{
		rng.seed(s);
	}


	//========================================================================
	/*!
	 *  \brief Assigns the pseudo random number generator instance "r" to the 
	 *         current random number generator instance.
	 *
	 *  \param r pseudo random number generator instance that will be assigned 
	 *           to the current instance
	 *  \return reference to the current pseudo random number generator instance
	 *          after the assignment
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
	RandomVar& operator = (const RandomVar& r)
	{
		rng = r.rng; return *this;
	}



	//========================================================================
	/*!
	 *  \brief For a distribution "p" given by the currently used random
	 *         number generator type, the entropy "H(p)" is calculated.
	 *
	 *  Say the currently used random number generator gives a 
	 *  distribution \f$p = \{p(x): x \in \chi\}\f$, where the number
	 *  of \f$x\f$ is determined by the number of Monte Carlo trials
	 *  preset by method #setMonteCarlo.
	 *  Then the entropy of \f$p\f$, \f$H(p)\f$ is given as
	 *
	 *  \f$
	 *      H(p) = - \sum_{x \in \chi} p(x) \cdot \log {p(x)} 
	 *           = E_p [- \log {p(X)}]
	 *  \f$
	 *
	 *  The entropy of a distribution can be thought of in some sense
	 *  as the \em surprise \em index of a random variable associated with
	 *  that distribution. <br>
	 *  Another way to see it is to think about the entropy as the average
	 *  number of yes/no questions needed to identify the outcome of a
	 *  random experience.
	 *
	 *  \return the entropy of the given distribution, \f$H(p)\f$
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
	 *  \sa #setMonteCarlo
	 *
	 */
	virtual double entropy()
	{
		double   t;
		unsigned i;

		for (t = 0, i = monteCarloTrials; i--;) {
			t += log(p((*this)()));
		}

		return -t / monteCarloTrials;
	}


	//========================================================================
	/*!
	 *  \brief Calculates the Kullback-Leibler distance for the two
	 *         distributions given by the type of random number generator
	 *         of the current instance and the generator given by "rv".
	 *
	 *  Say the currently used random number generator gives a 
	 *  distribution \f$p = \{p(x): x \in \chi\}\f$ and \em rv
	 *  gives the distribution \f$q = \{q(x): x \in \chi\}\f$, where the number
	 *  of \f$x\f$ is determined by the number of Monte Carlo trials
	 *  preset by method #setMonteCarlo. 
	 *  Then the Kullback-Leibler distance between \f$p\f$ and \f$q\f$, also 
	 *  known as \em divergence or \em relative \em entropy is given as
	 *
	 *  \f$
	 *      D(p,q) = \sum_{x \in \chi} p(x) \cdot \log {\frac{p(x)}{q(x)}} 
	 *           = E_p \big[ \log {\frac{p(X)}{q(X)}}\big] 
	 *           = E_p [ \mbox{log likelihood ratio} ]
	 *  \f$
	 *
	 *  \param rv the second random number generator used for the
	 *            distribution \f$q\f$.
	 *  \return the Kullback-Leibler distance between \f$p\f$ and \f$q\f$
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
	 *  \sa #setMonteCarlo
	 *
	 */
	virtual double KullbackLeibler(const RandomVar< T >& rv)
	{
		double   t;
		unsigned i;
		T        x;

		for (t = 0, i = monteCarloTrials; i--;) {
			x  = (*this)();
			t += log(p(x) / rv.p(x));
		}

		return t / monteCarloTrials;
	}

	//========================================================================
	/*!
	 *  \brief Calculate the log likelihood of the values in "x".
	 *
	 *  The log likelihood \f$l\f$ for a data vector \f$x\f$ with length \f$k\f$
	 *  is given as
	 *
	 *  \f$
	 *      l = \sum_{i=1}^k \log {P( x_i )}
	 *  \f$
	 *
	 *  where \f$P( x_i )\f$ is the probability of the occurrence of \f$x_i\f$.
	 *
	 *  \param x vector of data values for which the log likelihood will be
	 *           calculated
	 *  \return the log likelihood of the data vector \em x
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
	virtual double logLikelihood(const std::vector< T >& x) const
	{
		double l = 0;

		for (unsigned k = x.size(); k--;) {
			l += log(p(x[ k ]));
		}

		return l;
	}

	//========================================================================
	/*!
	 *  \brief Sets the maximum number of trials for Monte Carlo simulations.
	 *
	 *  The Monte Carlo simulation is an analytical method for the 
	 *  imitation of a real-life system when other analyses are too
	 *  mathematically complex or too difficult to produce. <br> 
	 *  Therefore the Monte Carlo simulation randomly generates values 
	 *  for uncertain variables over and over to simulate a model. <br>
	 *  The Monte Carlo simulation was named for Monte Carlo in Monaco
	 *  where the primary attractions are casinos containing games of
	 *  chance. The random behaviour in games of chance is similar to
	 *  how Monte Carlo simulation selects variable values at random
	 *  to simulate a model. In a casino you have the same situation
	 *  as in a Monte Carlo simulation, where you have variables that
	 *  have a known range of values but an uncertain value for any
	 *  particular time or event. <br>
	 *  For each uncertain variable you define the possible values
	 *  with a probability distribution. This distributions are defined
	 *  by the several types of random number generators defined in 
	 *  this library. <br>
	 *  Given a certain distribution, the simulation will calculate
	 *  multiple scenarios of a model by repeatedly sampling values from
	 *  the probability distributions for the uncertain variables. <br>
	 *  Setting the maximum number of Monte Carlo trials is used for
	 *  the definition of a distribution for the calculation of the #entropy 
	 *  and the calculation of the #KullbackLeibler distance for two
	 *  distributions, where one element of a distribution is chosen
	 *  by the type of random number generator.
	 *
	 *  \param N the maximum number of trials for the simulation, by default
	 *           set to "10000".
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
	 *  \sa #entropy, #KullbackLeibler
	 */
	void setMonteCarlo(unsigned N = 10000)
	{
		monteCarloTrials = N;
	}

protected:

	//! The pseudo random number generator taken from class
	//! RNG. This generator will produce uniformally distributed
	//! random numbers of the interval (0,1), that must be
	//! transformed to the distribution type of the derived class.
	RNG&     rng;

	//! The number of Monte Carlo Trials used for defining
	//! a distribution. For details see #setMonteCarlo,
	//! #entropy, #KullbackLeibler.
	unsigned monteCarloTrials;


	//========================================================================
	/*!
	 *  \brief Creates a new pseudo random generator instance, by using 
	 *         the generator "r" of class RNG and presets the number
	 *         of Monte Carlo trials to "10000" (see #setMonteCarlo).
	 *
	 *  Each instance of the template random number generator needs
	 *  a pseudo random number generator as defined
	 *  in class RNG. Normally this is only one single instance of
	 *  RNG named RNG::globalRng. To use the base methods for Monte
	 *  Carlo simulations (see #setMonteCarlo), the number of Monte
	 *  Carlo trials must be set, too.  
	 *
	 *  \param r the pseudo random number generator used for this
	 *           instance, by default this is the global generator 
	 *           RNG::globalRng
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
	 *  \sa #entropy, #KullbackLeibler
	 */
	RandomVar(RNG& r = RNG::globalRng)
			: rng(r),
			monteCarloTrials(10000)
	{}
};

#endif  /* !__RANDOMVAR_H */




