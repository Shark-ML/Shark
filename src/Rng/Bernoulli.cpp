//===========================================================================
/*!
 *  \file Bernoulli.cpp
 *
 *  \brief Implements methods for class Bernoulli that simulates a
 *         "Bernoulli trial", which is like a coin toss.
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
 *   <BR> 
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

#include "Rng/Bernoulli.h"


//========================================================================
/*!
 *  \brief Creates a new Bernoulli random generator instance and
 *         sets the probability for the occurrence of the event
 *         to "p".
 *
 *  The occurrence probability for the event is set to \em p, the
 *  probability for the non-occurrence is then \f$1 - p\f$. <br>
 *  For this instance, the default pseudo random number generator
 *  as member of class RandomVar is used.
 *
 *  \param p probability for the occurrence of the event
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
Bernoulli::Bernoulli(double p)
{
	pP = p;
}


//========================================================================
/*!
 *  \brief Creates a new %Bernoulli random generator instance by
 *         using the pseudo random number generator "r" for the determination
 *         of random values and sets the probability for the occurrence
 *         of the event to "p".
 *
 *  Each instance of a %Bernoulli random number generator is based
 *  on a generator, that is defined in class RNG and returns uniformally
 *  pseudo random numbers of the interval (0,1).
 *  By default, this is a global generator named RNG::globalRng and
 *  included as member in class RandomVar. <br>
 *  Here another pseudo random number generator \em r is used instead. <br>
 *  Additionally to defining the used pseudo random number generator,
 *  the occurrence probability for the %Bernoulli event is set to \em p, the
 *  probability for the non-occurrence is then \f$1 - p\f$. <br>
 *
 *  \param p probability for the occurrence of the event
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
Bernoulli::Bernoulli(double p, RNG& r)
		: RandomVar< bool >(r), pP(p)
{}


//========================================================================
/*!
 *  \brief Returns a Bernoulli random number, i.e. a "true" or "false"
 *         for the occurrence of an event when using the preset probability
 *         #pP.
 *
 *  \return "true", when the event occurred, "false" otherwise
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
bool Bernoulli::operator()()
{
	return operator()(pP);
}


//========================================================================
/*!
 *  \brief Returns the probability \f$p\f$ for the occurrence of an
 *         event ("x = true") or \f$1 - p\f$ for the non-occurrence
 *         ("x = false").
 *
 *  \param x if x is "true" then the probability \f$p\f$ for the
 *           occurrence of the event is returned, otherwise
 *           the probability \f$1 - p\f$ for the non-occurrence
 *  \return the probability for the (non-)occurrence of the event
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
double Bernoulli::p(const bool& x) const
{
	return x ? pP : 1 - pP;
}








