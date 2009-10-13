//===========================================================================
/*!
 *  \file RandomVector.h
 *
 *  \brief Vector valued random variable
 *
 *  \author  Martin Kreutz
 *  \date    1998-08-20
 *
 *  \par Copyright (c) 1995,2002:
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
 *      Mixture
 *
 *
 *      Revision 2.3  2007/09/19 11:38:44  leninga
 *      complete header
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Mixture. This library is free software;
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


#ifndef __RANDOMVECTOR_H
#define __RANDOMVECTOR_H


#include <SharkDefs.h>
#include <Array/Array.h>
#include <Rng/RandomVar.h>


//!
//! \brief Vector valued random variable
//!
template < class T >
class RandomVector : public RandomVar< Array< T > >
{
public:

	virtual ~RandomVector()
	{ }

	double logLikelihood(const Array< T >& x) const
	{
		SIZE_CHECK(x.ndim() == 2)

		double l = 0;

		for (unsigned k = x.dim(0); k--;) {
			l += log(Shark::max(p(x[ k ]), 1e-100));        // !!!
		}

		return l;
	}

protected:
	RandomVector(RNG& r = RNG::globalRng) : RandomVar< Array< T > > (r)
	{}
};


#endif

