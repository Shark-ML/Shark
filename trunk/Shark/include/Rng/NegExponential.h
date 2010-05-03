//===========================================================================
/*!
 *  \file NegExponential.h
 *
 *  \brief Contains a class that simulates a
 *         "negative exponential distribution".
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


#ifndef __NEGEXPONENTIAL_H
#define __NEGEXPONENTIAL_H

#include <cmath>
#include "Rng/RandomVar.h"


//===========================================================================
/*!
 *  \brief This class simulates a "negative exponential distribution".
 *
 *  This class is derived from class RandomVar and the uniformally
 *  distributed pseudo random number values of the interval (0,1)
 *  are transformed to type "double" of the negative exponential
 *  distribution (aka exponential distribution), that is given by: <br>
 *
 *  \f$
 *      f(x) = \lambda \cdot e^{- \lambda x}
 *  \f$
 *
 *  where \f$x \geq 0\f$ and \f$\lambda > 0\f$.
 *
 *  Below you can see the distribution for the parameter values
 *  \f$\lambda = 1.0\f$, \f$\lambda = 2.0\f$ and \f$\lambda = 4.0\f$:
 *
 *  \image html negExp.png
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
class NegExponential : public RandomVar< double >
{
public:

	//! Creates a new instance of the negative exponential random number
	//! generator and initializes the parameter \f$\lambda\f$.
	NegExponential(double mean = 0);

	//! Creates a new neg exponential random generator instance by
	//! using the pseudo random number generator "r" for the determination
	//! of random values and initializes the distribution's
	//! parameter \f$\lambda\f$.
	NegExponential(double mean, RNG& r);


	//========================================================================
	/*!
	 *  \brief Returns the current value of the parameter \f$\lambda\f$
	 *         as saved in #pMean.
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
	 *  \brief Sets the current value of the parameter \f$\lambda\f$
	 *         as saved in #pMean to the new value "newMean".
	 *
	 *  \param newMean value for #pMean (parameter \f$\lambda\f$),
	 *         that must be greater than "0", otherwise
	 *         \f$\lambda\f$ will not be changed
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      2002-04-16, ra: <br>
	 *      Now the parameter \f$\lambda\f$ is checked for
	 *      being greater than "0"
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void   mean(double newMean)
	{
		if (newMean > 0) pMean = newMean;
	}


//========================================================================
	/*!
	 *  \brief Returns a negative exponential random number for the
	 *         parameter \f$\lambda\f$ (#pMean) set to "mean".
	 *
	 *  This method performs the \em inverse \em transformation of the
	 *  original uniformally distributed random numbers of the interval
	 *  (0,1) created by the used pseudo random number generator to
	 *  the type of the negative exponential distribution.
	 *
	 *  \param mean the value for parameter \f$\lambda\f$ (#pMean), that
	 *              must be greater than zero, otherwise the random
	 *              number "0" will be returned
	 *  \return the negative exponential random number for
	 *          \f$\lambda = \mbox{mean}\f$ or "0" if \em mean is
	 *          less than or equal to zero
	 *
	 *  \author  M. Kreutz
	 *  \date    1998-08-17
	 *
	 *  \par Changes
	 *      2002-04-16, ra: <br>
	 *      Now the parameter \f$\lambda\f$ is checked for
	 *      being greater than "0"
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	double operator()(double mean)
	{
		if (mean > 0) return -mean * log(rng());
		else return 0.;
	}

	//! Returns a negative exponential random number for the
	//! parameter \f$\lambda\f$ as preset in #pMean.
	double operator()();

	//! Returns the probability for the occurrence of random number "x".
	double p(const double&) const;


protected:

	//! The parameter \f$\lambda\f$ in the distribution function
	//! (see class description for details).
	double pMean;
};

#endif  /* !__NEGEXPONENTIAL_H */






