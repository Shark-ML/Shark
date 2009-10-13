//===========================================================================
/*!
 *  \file Weibull.h
 *
 *  \brief Contains a class that simulates a "%Weibull distribution".
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


#ifndef __WEIBULL_H
#define __WEIBULL_H

#include <cmath>
#include "Rng/RandomVar.h"


//===========================================================================
/*!
 *  \brief This class simulates a "Weibull distribution".
 *
 *  This class is derived from class RandomVar and the uniformally
 *  distributed pseudo random number values of the interval (0,1)
 *  are transformed to type "double" of the %Weibull distribution. <br>
 *  The %Weibull distribution is an enhancement of the (Negative)
 *  %Exponential distribution to events that are not purely random
 *  and that can be not modelled exactly by the %Poisson distribution. <br>
 *  The %Weibull distribution is the classic reliability analysis
 *  and lifetime diagram. It is heavily used by the automobile
 *  and wind industry. <br>
 *  The distribution is given by
 *
 *  \f$
 *      f(x) = \frac{\alpha}{\beta} \cdot e^{-\frac{1}{\beta} \cdot x^\alpha} \cdot x^{\alpha - 1}
 *  \f$
 *
 *  with \f$x > 0\f$. <br>
 *  Below you can see the distribution for the parameter values
 *  \f$\alpha = 1\mbox{,\  } \beta = 2\f$,
 *  \f$\alpha = 2\mbox{,\  } \beta = 1\f$ and
 *  \f$\alpha = 2\mbox{,\  } \beta = 3\f$:
 *
 *  \image html weibull.png
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
class Weibull : public RandomVar< double >
{
public:

	//! Creates a new instance of the %Weibull random number
	//! generator and initializes the distribution's parameters.
	Weibull(double alpha = 1, double beta = 1);

	//! Creates a new %Weibull random generator instance by
	//! using the pseudo random number generator "r" for the determination
	//! of random values and initializes the distribution's parameters.
	Weibull(double alpha, double beta, RNG& r);


//========================================================================
	/*!
	 *  \brief Returns the current value of the distribution's
	 *         parameter \f$\alpha\f$.
	 *
	 *  \return the current value of \f$\alpha\f$ as stored in #pAlpha
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
	double alpha() const
	{
		return pAlpha;
	}


//========================================================================
	/*!
	 *  \brief Returns the current value of the distribution's
	 *         parameter \f$\beta\f$.
	 *
	 *  \return the current value of \f$\beta\f$ as stored in #pBeta
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
	double beta() const
	{
		return pBeta;
	}


//========================================================================
	/*!
	 *  \brief Sets the distribution's parameter \f$\alpha\f$
	 *         to the new value "a".
	 *
	 *  \param  a the new value for parameter \f$\alpha\f$ as stored in #pAlpha
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
	void   alpha(double a)
	{
		pAlpha = a;
	}


//========================================================================
	/*!
	 *  \brief Sets the distribution's parameter \f$\beta\f$
	 *         to the new value "b".
	 *
	 *  \param  b the new value for parameter \f$\beta\f$ as stored in #pBeta
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
	void   beta(double b)
	{
		pBeta  = b;
	}


//========================================================================
	/*!
	 *  \brief For the given distribution parameters "alpha" and "beta"
	 *         this method returns a %Weibull random number.
	 *
	 *  This method performs the \em inverse \em transformation of the
	 *  original uniformally distributed random numbers of the interval
	 *  (0,1) created by the used pseudo random number generator to
	 *  the type of the %Weibull distribution.
	 *
	 *  \param  alpha the distribution's parameter \f$\alpha\f$
	 *  \param  beta  the distribution's parameter \f$\beta\f$
	 *  \return a %Weibull random number
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
	double operator()(double alpha, double beta)
	{
		return pow(-beta * log(1 - rng()), 1 / alpha);
	}

	//! For the current distribution parameters \f$\alpha\f$ and
	//! \f$\beta\f$, this method returns a %Weibull random number.
	double operator()();

	//! Returns the probability for the occurrence of random
	//! number "x" for the %Weibull distribution with the
	//! parameter values \f$\alpha\f$ as stored in #pAlpha and
	//! \f$\beta\f$ as stored in #pBeta.
	double p(const double&) const;

	//! Returns the probability for the occurrence of random
	//! number "x" for the %Weibull distribution with the
	//! values "a" for parameter \f$\alpha\f$ and "b" for
	//! parameter \f$\beta\f$.
	double p(const double &, const double &, const double &) const;


protected:

	//! The distribution's parameter \f$\alpha\f$ (see class description).
	double pAlpha;


	//! The distribution's parameter \f$\beta\f$ (see class description).
	double pBeta;


};

#endif /* !__WEIBULL_H */



