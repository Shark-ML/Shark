//===========================================================================
/*!
 *  \file DiffGeometric.cpp
 *
 *  \brief Implements methods for class DiffGeometric that simulates a
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

#include "Rng/DiffGeometric.h"


//========================================================================
/*!
 *  \brief For a given probability "mean" for a single %Benoulli trial,
 *         this method returns a differential geometric
 *         random number as difference between two geometric random
 *         numbers (trial difference).
 *
 *  \return the differential geometric random number (trial difference)
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
long DiffGeometric::operator()(double mean)
{
	return Geometric::operator()(mean) - Geometric::operator()(mean);
}



//========================================================================
/*!
 *  \brief For the current probability for a single %Benoulli trial stored
 *         in Geometric::pMean, this method returns a differential geometric
 *         random number as difference between two geometric random
 *         numbers (trial difference).
 *
 *  \return the differential geometric random number (trial difference)
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
long DiffGeometric::operator()()
{
	return Geometric::operator()() - Geometric::operator()();
}


//========================================================================
/*!
 *  \brief For a value "x" of trial difference, this method returns the
 *         probability as given by the distribution function.
 *
 *  \param x the trial difference
 *  \return the probability, that the trial difference is \em x
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
double DiffGeometric::p(const long& x) const
{
	return pMean * pow(1 - pMean, abs(x)) / (2 - pMean);
}



