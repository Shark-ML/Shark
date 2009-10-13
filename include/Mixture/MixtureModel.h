//===========================================================================
/*!
 *  \file MixtureModel.h
 *
 *  \brief Abstract model of a parameterized mixture density
 *
 *  \author  Martin Kreutz
 *  \date    1998-09-06
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

#ifndef __MIXTUREMODEL_H
#define __MIXTUREMODEL_H

#include <SharkDefs.h>
#include <Mixture/RandomVector.h>
#include <Array/ArrayOp.h>


//!
//! \brief Abstract model of a parameterized mixture density
//!
template < class T >
class MixtureModel : public RandomVector< T >
{
public:

	virtual ~MixtureModel()
	{ }

	virtual double p(const Array< T >&, unsigned) const = 0;

	virtual void resize(unsigned n, bool copy = true)
	{
		a.resize(n, copy);
	}

	virtual void normalize()
	{
		double suma = sum(a);
		//FLOAT_CHECK( suma )
		if (suma != 0)
			a /= suma;
	}

	virtual void initialize()
	{
		double sum = 0;
		for (unsigned i = a.nelem(); i--; sum += (a(i) = uni()));
		a /= sum;
	}

	double p(const Array< T >& x) const
	{
		double px = 0;

		for (unsigned i = size(); i--;) {
			px += a(i) * p(x, i);
		}

		//FLOAT_CHECK( px )
		//RANGE_CHECK( px >= 0 )
		return px;
	}

	double p(unsigned i, const Array< T >& x) const
	{
		//  return a( i ) * p( x, i ) / p( x );
		double pxi = a(i) * p(x, i);
		double pix = 0.;

		for (unsigned j = size(); j--;)
			pix += j == i ? pxi : a(j) * p(x, j);

		//FLOAT_CHECK( pxi )
		//FLOAT_CHECK( pix )
		//RANGE_CHECK( pxi >= 0 )
		//RANGE_CHECK( pix >= 0 )
		return pxi > 0 ? pxi / pix : 0;
	}

	double p(unsigned i) const
	{
		return a(i);
	}

	//
	// select a model randomly equal probability
	//
	unsigned sampleModelUniform()
	{
		return unsigned(uni() * size());
	}

	//
	// select a model randomly with p( i )
	//
	unsigned sampleModel()
	{
		unsigned i;
		double   u = uni();
		double   s = 0;

		for (i = 0; i < size() && u >= a(i) + s; s += a(i++));
		if (i == size()) i = size() - 1;    // should never happen

		return i;
	}

	//
	// select a model randomly with p( i | x )
	//
	unsigned sampleModel(const Array< T >& x)
	{
		unsigned i;
		double   pi;
		double   u = uni();
		double   s = 0;

		for (i = 0;
				i < size() && u > (pi = p(i, x)) + s;
				s += pi, ++i);
		if (i == size()) i = size() - 1;    // should never happen

		return i;
	}

	double modelEntropy(const Array< T >& x) const
	{
		unsigned i, k;
		double   px, h;
		Array< double > pxi(size());
		Array< double > s(size());

		//SIZE_CHECK( x.ndim( ) == 1 || x.ndim( ) == 2 )

		if (x.ndim() == 2) {
			for (s = 0, k = 0; k < x.dim(0); ++k) {
				for (px = 0, i = 0; i < size(); ++i)
					px += (pxi(i) = a(i) * p(x[ k ], i));
				for (i = 0; i < size(); ++i)
					s(i) += pxi(i) / Shark::max(px, /*MIN_VAL !!!*/ 1e-100);
			}
			s /= double(x.dim(0));
		}
		else {
			for (px = 0, i = 0; i < size(); ++i)
				px += (pxi(i) = a(i) * p(x, i));
			for (i = 0; i < size(); ++i)
				s(i) = pxi(i) / px;
		}

		for (h = 0, i = 0; i < size(); ++i)
			if (s(i) > /*MIN_VAL!!!*/ 1e-100)
				h -= s(i) * log(s(i));

		return h;
	}

	double& prior(unsigned i)
	{
		return a(i);
	}
	double  prior(unsigned i) const
	{
		return a(i);
	}
	const Array< double >& prior() const
	{
		return a;
	}

	unsigned size() const
	{
		return a.nelem();
	}

protected:
	Uniform uni;
	Array< double > a;

	MixtureModel(unsigned size = 0, RNG& r = RNG::globalRng)
			: RandomVector< T >(r), uni(0, 1, r), a(size)
	{
		if (size) a = 1. / size;
	}
};

#endif  /* !__MIXTUREMODEL_H */

