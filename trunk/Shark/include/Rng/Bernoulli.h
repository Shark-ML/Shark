//===========================================================================
/*!
 *  \file Bernoulli.h
 *
 *  \brief Contains a class that simulates a "Bernoulli trial", which
 *         is like a coin toss.
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


#ifndef __BERNOULLI_H
#define __BERNOULLI_H

#include "Rng/RandomVar.h"


//===========================================================================
/*!
 *  \brief This class simulates a "Bernoulli trial", which
 *         is like a coin toss.
 *
 *  This class is derived from class RandomVar and the uniformally
 *  distributed pseudo random number values of the interval (0,1)
 *  are transformed to type "bool", because an event
 *  (side of a coin) can occur or can not occur. <br>
 *  Therefore a probability for the occurrence of the event (coin side)
 *  is defined. When using the equal probability of "0.5" for the
 *  occurrence and non-occurrence of the event (coin side), then the
 *  event (coin) is named "normal", otherwise it is named "abnormal".
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
 */
class Bernoulli : public RandomVar< bool >
{
public:

	//! Creates a new Bernoulli random generator instance and
	//! sets the probability for the occurrence of the event
	//! to "p".
	Bernoulli(double p = 0.5);

	//! Creates a new Bernoulli random generator instance by
	//! using the pseudo random number generator "r" for the determination
	//! of random values and sets the probability for the occurrence
	//! of the event to "p".
	Bernoulli(double p, RNG& r);



//========================================================================
	/*!
	 *  \brief Returns the probability for the occurrence of an event.
	 *
	 *  \return the probability #pP for the occurrence of an event
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
	double prob() const
	{
		return pP;
	}


//========================================================================
	/*!
	 *  \brief Sets the probability for the occurrence of an event to "newP".
	 *
	 *  \param newP the new probability for the occurrence #pP of an event
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
	void   prob(double newP)
	{
		pP = newP;
	}


//========================================================================
	/*!
	 *  \brief Returns a Bernoulli random number, i.e. a "true" or "false"
	 *         marking the occurrence and non-occurrence of an event respectively,
	 *         when the probability for the occurrence is "p".
	 *
	 *  This method performs the \em inverse \em transformation of the
	 *  original uniformally distributed random numbers of the interval
	 *  (0,1) created by the used pseudo random number generator to
	 *  the type of the Bernoulli distribution.
	 *
	 *  \param p probability for the occurrence of the event
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
	bool   operator()(double p)
	{
		return rng() < p;
	}

	//! Returns a Bernoulli random number, i.e. a "true" or "false"
	//! for the occurrence of an event when using the preset probability
	//! #pP.
	bool operator()();


	//! Returns the probability \f$p\f$ for the occurrence of an
	//! event ("x = true") or \f$1 - p\f$ for the non-occurrence
	//! ("x = false").
	double p(const bool&) const;

protected:

	//! The probability \f$p\f$ for the occurrence of the event (coin side).
	//! The probability for the non-occurrence of the event is then
	//! given by \f$1 - pP\f$.
	double pP;
};

#endif /* !__BERNOULLI_H */


