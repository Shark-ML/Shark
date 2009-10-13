//===========================================================================
/*!
*  \file Dirichlet.h
*
*  \brief Dirichlet distribution
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


#ifndef __DIRICHLET_H
#define __DIRICHLET_H

#include <cmath>

#include "SharkDefs.h"
#include "Rng/RandomVar.h"
#include "Rng/Gamma.h"


//===========================================================================
/*!
*  \brief This class simulates a "Dirichlet" distribution.
*
*  The Dirichlet distribution of order \f$K-1\f$ is an
*  \f$K\f$-dimensional continous distribution with \f$K\f$ positive
*  parameters \f$\alpha_1,\dots,\alpha_K\f$.  The probability density
*  function satisfies \f$ f(x_1,\dots,x_K)\propto \prod_{i=1}^K
*  x_i^{\alpha_i-1}\f$ and the \f$x_i\f$ are positive and sum to one.

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
class Dirichlet : public RandomVar< std::vector<double> >
{
public:

	//! Creates a new instance of the Dirichlet random number
	//! generator
	//! \param n length of random vector
	//! \param a common parameter value 
	Dirichlet(unsigned n=3, double alpha=1);

	//! Creates a new instance of the Dirichlet random number
	//! generator
	//! \param a parameter vector 
	Dirichlet(const std::vector<double> &alpha);

	//! Creates a new instance of the Dirichlet random number
	//! generator
	//! \param n length of random vector 
	//! \param a common parameter value 
	//! \param r random number generator 
  Dirichlet(unsigned n, double alpha, RNG& r);

	//! Creates a new instance of the Dirichlet random number
	//! generator
	//! \param a parameter vector 
	//! \param r random number generator 
  Dirichlet(std::vector<double> &alpha, RNG& r);

	//========================================================================
	/*!
	 *  \brief Initializes the pseudo random number generator used by this 
	 *         class with value \em s.
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

	//! Set parameter vector.
	void   alpha(const std::vector<double> &a);

	//! Set each parameter to the scalar value \em a.
	void   alpha(double a);

	//========================================================================
	/*!
	 *  \brief Samples Dirichlet distribution with parameters \em  a
	 *
	 *  The dimensionality of the random vector corresponds to the
	 *  dimensionality of the parameter vector \em  a.
	 *
	 *  \param a parameter vector 
	 *  \return realization of random vector
	 *
	 *  \author  C. Igel
	 *  \date    2008-11-28
	 *
	 *  \sa RandomVar::seed
	 *
	 */	
	std::vector <double> operator()(const std::vector <double> &alpha);

	//========================================================================
	/*!
	 *  \brief Samples Dirichlet distribution where all parameters are
	 *  set to the same scalar \em a
	 *
	 *  The dimensionality of the random vector corresponds to the
	 *  parameter \em  n
	 *
	 *  \param n length of random vector 
	 *  \param a common parameter value  
   *
	 *  \return realization of random vector
	 *
	 *  \author  C. Igel
	 *  \date    2008-11-28
	 *
	 *  \sa RandomVar::seed
	 *
	 */
	std::vector <double> operator()(unsigned n, double alpha);

	//! Returns a Dirichlet distributed random number 
	std::vector <double> operator()();

	//! Returns the probability desity function evaluated at \em x
	double p(const std::vector<double> &x) const;

protected:
	//! Parameter vector
	std::vector<double > pAlpha;

};

#endif  /* !__DIRICHLET_H */
