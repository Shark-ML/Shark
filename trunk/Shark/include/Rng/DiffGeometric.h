//===========================================================================
/*!
 *  \file DiffGeometric.h
 *
 *  \brief Contains a class that simulates a
 *         "Differential Geometric distribution".
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

#ifndef __DIFFGEOMETRIC_H
#define __DIFFGEOMETRIC_H

#include "Rng/Geometric.h"


//===========================================================================
/*!
 *  \brief This class simulates a "Differential %Geometric distribution".
 *
 *  This class is derived from class Geometric and uses two geometric random
 *  numbers based on the same probability for a single Bernoulli
 *  trial. The differential geometric random number is then
 *  the subtraction of the two geometric random numbers, so you can
 *  interpret it as trial difference. <br>
 *  In contrast to class Geometric, the distribution function is here
 *  given by:
 *
 *  \f$
 *      f(x) = ( p \cdot (1 - p)^{|x|} ) / (2 - p)
 *  \f$
 *
 *  where \f$p\f$ is the probability for a single %Bernoulli trial as stored
 *  in Geometric::pMean. <br>
 *  Below you can see the distribution function for the probabilities
 *  \f$p = 0.2\f$, \f$p = 0.5\f$ and \f$p = 0.8\f$ for a single %Bernoulli
 *  trial:
 *
 *  \image html diffGeom.png
 *  \image latex diffGeom.eps
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
class DiffGeometric : public Geometric
{
public:


//========================================================================
	/*!
	 *  \brief Creates a new instance of the differential geometric random number
	 *         generator and sets the probability for a single
	 *         %Bernoulli trial.
	 *
	 *  The probability Geometric::pMean for a single %Bernoulli trial is set
	 *  to \em mean.
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
	DiffGeometric(double mean = 0) : Geometric(mean)
	{ }


//========================================================================
	/*!
	 *  \brief Creates a new differential geometric random generator instance by
	 *         using the pseudo random number generator "r" for the determination
	 *         of random values and sets the probability for a single
	 *         %Bernoulli trial.
	 *
	 *  Each instance of a differential geometric random number generator is based
	 *  on a generator, that is defined in class RNG and returns uniformally
	 *  pseudo random numbers of the interval (0,1).
	 *  By default, this is a global generator named RNG::globalRng and
	 *  included as member in class RandomVar. <br>
	 *  Here another pseudo random number generator \em r is used instead. <br>
	 *  Additionally to defining the used pseudo random number generator,
	 *  the probability Geometric::pMean for a single %Bernoulli trial is
	 *  set to \em mean.
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
	DiffGeometric(double mean, RNG& r) : Geometric(mean, r)
	{ }

	//! For a given probability "mean" for a single %Benoulli trial,
	//! this method returns a differential geometric
	//! random number as difference between two geometric random
	//! numbers (trial difference).
	long operator()(double mean);

	//! For the current probability for a single %Benoulli trial stored
	//! in Geometric::pMean, this method returns a differential geometric
	//! random number as difference between two geometric random
	//! numbers (trial difference).
	long operator()();

	//! For a value "x" of trial difference, this method returns the
	//! probability as given by the distribution function.
	double p(const long&) const;

};

#endif  /* !__DIFFGEOMETRIC_H */

