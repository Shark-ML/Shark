//===========================================================================
/*!
 *  \file GlobalRng.h
 *
 *  \brief Contains a class that subsumes several often used random number
 *         generators.
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


#ifndef __GLOBALRNG_H
#define __GLOBALRNG_H

#include "Rng/Bernoulli.h"
#include "Rng/DiscreteUniform.h"
#include "Rng/Uniform.h"
#include "Rng/Normal.h"
#include "Rng/Cauchy.h"
#include "Rng/Geometric.h"
#include "Rng/DiffGeometric.h"
#include "Rng/Poisson.h"
#include "Rng/Gamma.h"
#include "Rng/Dirichlet.h"


//===========================================================================
/*!
 *  \brief This class subsumes several often used random number generators.
 *
 *  This class instantiates the following classes at once:
 *
 *  <ul>
 *      <li>Bernoulli with name \em coinToss
 *      <li>DiscreteUniform with name \em discrete
 *      <li>Uniform with name \em uni
 *      <li>Normal with name \em gauss
 *      <li>Cauchy with name \em cauchy
 *      <li>Geometric with name \em geom
 *      <li>DiffGeometric with name \em diffGeom
 *      <li>Poisson with name \em poisson
 *      <li>Gamma with name \em gam
 *      <li>Dirichlet with name \em dir
 *  </ul>
 *
 *  Additionally, the seed for all the random number generators listed above
 *  can be set by calling a single method.
 *
 *  \par Example
 *  \code
 *  #include "Rng/GlobalRng.h"
 *
 *  void main()
 *  {
 *      // We need only one instance to get several
 *      // random number generators:
 *      Rng rng;
 *
 *      // Set seed for all subsumed random number generators:
 *      rng.seed( 1234 );
 *
 *      // Get random "numbers" for all subsumed random number generators:
 *      bool   rn1 = rng.coinToss( );
 *      long   rn2 = rng.discrete( );
 *      double rn3 = rng.uni( );
 *      double rn4 = rng.gauss( );
 *      double rn5 = rng.cauchy( );
 *      long   rn6 = rng.geom( );
 *      long   rn7 = rng.diffGeom( );
 *
 *      // Output of random numbers:
 *      cout << "Bernoulli trial                              = " << rn1 << endl;
 *      cout << "Discrete distribution number                 = " << rn2 << endl;
 *      cout << "Uniform distribution number                  = " << rn3 << endl;
 *      cout << "Normal distribution number                   = " << rn4 << endl;
 *      cout << "Cauchy distribution number                   = " << rn5 << endl;
 *      cout << "Geometric distribution number                = " << rn6 << endl;
 *      cout << "Differential Geometric distribution number   = " << rn7 << endl;
 *  }
 *  \endcode
 *
 *  This is the way you can use the class and address the single
 *  random number generators in it.
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
class Rng
{
public:

	//! Instance of class Bernoulli:
	static Bernoulli       coinToss;

	//! Instance of class DiscreteUniform:
	static DiscreteUniform discrete;

	//! Instance of class Uniform:
	static Uniform         uni;

	//! Instance of class Normal:
	static Normal          gauss;

	//! Instance of class Cauchy:
	static Cauchy          cauchy;

	//! Instance of class Geometric:
	static Geometric       geom;

	//! Instance of class DiffGeometric:
	static DiffGeometric   diffGeom;

	//! Instance of class Poisson:
	static Poisson         poisson;

	//! Instance of class Gamma:
	static Gamma           gam;

	//! Instance of class Dirichlet:
	static Dirichlet       dir;

	//! Sets the seed for all random number generators
	//! to "s".
	static void seed(long s);
};

#endif /* !__GLOBALRNG_H */



