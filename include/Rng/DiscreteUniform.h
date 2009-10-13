//===========================================================================
/*!
 *  \file DiscreteUniform.h
 *
 *  \brief Contains a class that simulates a "uniform distribution"
 *         with integer numbers.
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


#ifndef __DISCRETEUNIFORM_H
#define __DISCRETEUNIFORM_H

#include "Rng/RandomVar.h"


//===========================================================================
/*!
 *  \brief This class simulates a "uniform distribution" with integer
 *         numbers.
 *
 *  This class is derived from class RandomVar and the uniformally
 *  distributed pseudo random number values of the interval (0,1)
 *  are transformed to type "long" and are from an interval
 *  \f$[\f$#pLow, #pHigh\f$[\f$. <br>
 *  This class is similar to class Uniform, but here the distribution
 *  is not continuous but discrete, so you will receive integer random
 *  numbers instead of double random numbers.
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
class DiscreteUniform : public RandomVar< long >
{
public:


	//! Creates a new instance of the discrete uniform random number
	//! generator and initializes the lower and upper bound.
	DiscreteUniform(long lo = 0, long hi = 1);


	//! Creates a new discrete uniform random generator instance by
	//! using the pseudo random number generator "r" for the determination
	//! of random values and initializes the interval bounds
	//! for the random numbers.
	DiscreteUniform(long lo, long hi, RNG& rng);


//========================================================================
	/*!
	 *  \brief Returns the lower (included) bound #pLow for the random
	 *         numbers interval.
	 *
	 *  \return the lower bound #pLow of the interval
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
	long low() const
	{
		return pLow;
	}


//========================================================================
	/*!
	 *  \brief Returns the upper (excluded) bound #pHigh for the random
	 *         numbers interval.
	 *
	 *  \return the upper bound #pHigh of the interval
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
	long high() const
	{
		return pHigh;
	}


//========================================================================
	/*!
	 *  \brief Sets the lower (included) bound #pLow for the random
	 *         numbers interval to the new value "lo".
	 *
	 *  \param lo the new value for the lower bound #pLow
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
	void low(long lo)
	{
		pLow  = lo;
	}


//========================================================================
	/*!
	 *  \brief Sets the upper (excluded) bound #pHigh for the random
	 *         numbers interval to the new value "hi".
	 *
	 *  \param hi the new value for the upper bound #pHigh
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
	void high(long hi)
	{
		pHigh = hi;
	}


//========================================================================
	/*!
	 *  \brief Returns a uniformally distributed discrete random number
	 *         from the interval ["lo", "hi"].
	 *
	 *  This method performs the \em inverse \em transformation of the
	 *  original uniformally distributed random numbers of the interval
	 *  (0,1) created by the used pseudo random number generator to
	 *  the type of the discrete uniform distribution, i.e.
	 *  a discrete random number \f$rn\f$ with \f$lo \leq rn \leq hi\f$
	 *  is returned.
	 *
	 *  \param lo the minimum random number that can be returned
	 *  \param hi the upper bound for random numbers that can be returned
	 *  \return the discrete random number \f$rn\f$
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
	long operator()(long lo, long hi)
	{
		return lo + long(rng() *(hi - lo + 1));
	}


	//! Returns a uniformally distributed discrete random number
	//! from the interval [#pLow, #pHigh].
	long operator()();


	//! Returns the probability for the occurrence of random number "x".
	double p(const long&) const;

protected:

	//! The lower bound of the random number interval
	//! \f$[\f$#pLow, #pHigh\f$]\f$.
	long pLow;

	//! The upper bound of the random number interval
	//! \f$[\f$#pLow, #pHigh\f$]\f$.
	long pHigh;
};

#endif /* !__DISCRETEUNIFORM_H */

