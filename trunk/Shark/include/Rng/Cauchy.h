//===========================================================================
/*!
 *  \file Cauchy.h
 *
 *  \brief Contains a class that simulates a "standard %Cauchy distribution"
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


#ifndef __CAUCHY_H
#define __CAUCHY_H

#include "Rng/RandomVar.h"


//===========================================================================
/*!
 *  \brief This class simulates a "standard %Cauchy distribution".
 *
 *  This class is derived from class RandomVar and the uniformally
 *  distributed pseudo random number values of the interval (0,1)
 *  are transformed to type "double" of distribution %Cauchy. <br>
 *  The %Cauchy distribution (aka "Lorentzian") is defined by:
 *
 *  \f$
 *      f(x) = \frac{1}{\pi (1 + x^2)}
 *  \f$
 *
 *  Below you can see the distribution:
 *
 *  \image html cauchy.png
 *  \image latex cauchy.eps
 *
 *  <br>
 *  The %Cauchy distribution is important as an example of a pathological
 *  case. The %Cauchy distribution looks similar to a Normal distribution,
 *  but has much heavier tails. When studying hypothesis tests that assume
 *  normality, seeing how the tests perform on data from a %Cauchy
 *  distribution is a good indicator of how sensitive the tests are to
 *  heavy-tail departures from normality. Likewise, it is a good check
 *  for robust techniques that are designed to work well under a wide
 *  variety of distributional assumptions.
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
class Cauchy : public RandomVar< double >
{
public:


//========================================================================
	/*!
	 *  \brief Creates a new %Cauchy random generator instance.
	 *
	 *  For this instance, the default pseudo random number generator
	 *  as member of class RandomVar is used.
	 *
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
	Cauchy()
	{ }


//========================================================================
	/*!
	 *  \brief Creates a new %Cauchy random generator instance by
	 *         using the pseudo random number generator "r" for the determination
	 *         of random values.
	 *
	 *  Each instance of a %Cauchy random number generator is based
	 *  on a generator, that is defined in class RNG and returns uniformally
	 *  pseudo random numbers of the interval (0,1).
	 *  By default, this is a global generator named RNG::globalRng and
	 *  included as member in class RandomVar. <br>
	 *  Here another pseudo random number generator \em r is used instead.
	 *
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
	Cauchy(RNG& r) : RandomVar< double >(r)
	{ }

	//! Returns a %Cauchy random number.
	double operator()();

	//! Returns the probability for the occurrence of random number "x".
	double p(const double&) const;

};

#endif  /* !__CAUCHY_H */




