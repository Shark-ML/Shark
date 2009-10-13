//===========================================================================
/*!
*  \file Gamma.h
*
*  \brief Gamma distribution
*
*  \author  C. Igel
*  \date    2008-11-28
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
*  <BR>
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


#ifndef __GAMMA_H
#define __GAMMA_H

#include <cmath>
#include "SharkDefs.h"
#include "Rng/RandomVar.h"


//===========================================================================
/*!
*  \brief This class simulates a "gamma" distribution.
*
*  The gamma distribution is a two-parameter family of continuous
*  probability distributions having a scale parameter \f$\theta\f$ and a shape
*  parameter \f$k\f$. Samples from the gamma distribution are positive.<p>
*
* The  probability density function is 
* \f$
* f(x)=x^{k-1} \frac{\exp{\left(-x/\theta\right)}}{\Gamma(k)\,\theta^k}
* \f$
* and the mean is \f$\theta k\f$ and the variance \f$\theta^2 k\f$.
*
*  \author  C. Igel
*  \date    2008-11-28
*
*  \par Changes:
*      none
*
*  \par Status:
*      stable
*
*/
class Gamma : public RandomVar< double >
{
public:

	//! Creates a new instance of the gamma random number
	//! generator, initializes the shape and scale parameter
	Gamma(double k = 1, double theta = 1);


	//! Creates a new gamma random generator instance by
	//! using the pseudo random number generator \em r
	//! nd initializes the range of the numbers and
	//! internal variables.
	Gamma(double k, double theta, RNG& r);

	//========================================================================
	/*!
	 *  \brief Initializes the pseudo random number generator used by this 
	 *         class with value \em s
	 *
	 *  The pseudo random number generator as defined in class RNG is
	 *  initialized by using the seed value \em s. <br>
	 *
	 *  \param s initialization value for the pseudo random number generator
	 *  \return none
	 *
	 *  \author  C. Igel
	 *  \date    2008-11-28
	 *
	 *  \sa RandomVar::seed
	 *
	 */
	void seed(long s);


	//========================================================================
	/*!
	 *  \brief Returns the current mean value \f$k \theta\f$ 
	 *
	 *  \return the mean value 
	 *
	 *  \author  C. Igel
	 *  \date    1995-01-01
	 *
	 */
	double mean() const;


	//========================================================================
	/*!
	 *  \brief Returns the current variance \f$ k\theta^2\f$
	 *
	 *  \return the variance 
	 *
	 *  \author  C. Igel
	 *  \date    2008-11-28
	 *
	 */
	double variance() const;


	//========================================================================
	/*!
	 *  \brief Sets the current shape parameter
	 *
	 *  \param s new shape
	 *  \return none
	 *
	 *  \author  C. Igel
	 *  \date    2008-11-28
	 *
	 */
	void   shape(double s);

	//========================================================================
	/*!
	 *  \brief Sets the current scale parameter of the distribution
	 *
	 *  \param  s new scale 
	 *  \return none
	 *
	 *  \author  C. Igel
	 *  \date    2008-11-28
	 *
	 */
	void   scale(double s);

	//========================================================================
	/*!
	 *  \brief Returns gamma distributed random number
	 *
	 *  \param  k shape 
	 *  \param  theta scale 
	 *  \return none
	 *
	 *  \author  C. Igel
	 *  \date    2008-11-28
	 *
	 */
	double operator()(double k, double theta);

	//! Returns a gamma distributed random number 
	double operator()();

	//! Returns the probability desity function evaluated at \em x
	double p(const double &x) const;

protected:

	//! Shape parameter \f$k\f$ of the gamma distribution.
	double pK;

	//! Scale parameter \f$\theta\f$ of the distribution.
	double pTheta;

	// drawing uniformaly from ]0,1]
	double U01() { return 1.-rng(); }

};

#endif  /* !__GAMMA_H */
