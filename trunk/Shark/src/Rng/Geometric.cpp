//===========================================================================
/*!
 *  \file Geometric.cpp
 *
 *  \brief Implements methods for class Geometric that simulates a
 *         "%Geometric distribution".
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


#include <cmath>
#include "Rng/Geometric.h"


//========================================================================
/*!
 *  \brief Creates a new instance of the geometric random number
 *         generator and sets the probability for a single
 *         %Bernoulli trial.
 *
 *  The probability #pMean for a single %Bernoulli trial is set to \em mean.
 *  <br>
 *  For this instance, the default pseudo random number generator
 *  as member of class RandomVar is used.
 *
 *  \param mean     the probability for a single %Bernoulli trial,
 *                  the default is "0"
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
Geometric::Geometric(double mean)
		: pMean(mean)
{}

//========================================================================
/*!
 *  \brief Creates a new geometric random generator instance by
 *         using the pseudo random number generator "r" for the determination
 *         of random values and sets the probability for a single
 *         %Bernoulli trial.
 *
 *  Each instance of a geometric random number generator is based
 *  on a generator, that is defined in class RNG and returns uniformally
 *  pseudo random numbers of the interval (0,1).
 *  By default, this is a global generator named RNG::globalRng and
 *  included as member in class RandomVar. <br>
 *  Here another pseudo random number generator \em r is used instead. <br>
 *  Additionally to defining the used pseudo random number generator,
 *  the probability #pMean for a single %Bernoulli trial is set to \em mean.
 *
 *  \param mean     the probability for a single %Bernoulli trial
 *  \param r the pseudo random number generator that is used
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
Geometric::Geometric(double mean, RNG& r)
		: RandomVar< long >(r), pMean(mean)
{}


//========================================================================
/*!
 *  \brief For a given probability "mean" for a single %Benoulli trial,
 *         this method returns the number of trials after which the
 *         first success occurs.
 *
 *  This method performs the \em inverse \em transformation of the
 *  original uniformally distributed random numbers of the interval
 *  (0,1) created by the used pseudo random number generator to
 *  the type of the geometric distribution.
 *
 *  \param mean     the probability for a single %Bernoulli trial
 *  \return the number of trials after which the first success occurs
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
long Geometric::operator()(double mean)
{
	return long(ceil(log(1 - rng()) / log(1 - mean)));
}


//========================================================================
/*!
 *  \brief For the current probability for a single %Benoulli trial stored
 *         in #pMean, this method returns the number of trials after which the
 *         first success occurs.
 *
 *  \return the number of trials after which the first success occurs
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
long Geometric::operator()()
{
	return operator()(pMean);
}


//========================================================================
/*!
 *  \brief For a number of trials "x", this method returns the
 *         probability that the first success occurs after this
 *         number of trials.
 *
 *  \param x the number of trials after which the first success shall occur
 *  \return the probability, that the first success occurs after
 *          \em x trials
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
double Geometric::p(const long& x) const
{
	return x > 0 ? pMean * pow(1 - pMean, x - 1) : 0;
}

