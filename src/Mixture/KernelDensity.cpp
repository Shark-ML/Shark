//===========================================================================
/*!
 *  \file KernelDensity.cpp
 *
 *  \brief Class for kernel density estimators
 *
 *  \author  Martin Kreutz
 *  \date    1995-01-01
 *
 *  \par Copyright (c) 1995,2002:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>

 *  \par Project:
 *      Mixture
 *
 *
 *
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
 *
 */
//===========================================================================

#include <SharkDefs.h>
#include <Rng/GlobalRng.h>
#include <Array/ArrayOp.h>
#include <Mixture/KernelDensity.h>

//===========================================================================

Array< double > KernelDensity::sample()
{
	unsigned i, j;
	Array< double > s(dim());

	//
	// select gaussian
	//
	i = unsigned(Rng::uni(0, size()));

	//
	// sample from selected gaussian
	//
	for (j = dim(); j--;) {
		s(j) = Rng::gauss(m(i, j), var);
	}

	return s;
}

//===========================================================================

// 2003-04-14:
// Added new calculation of "p" for case "var == 0.", because
// in this case the value "NaN" was returned.
// Idea of the new calculation: features with discrete values
// are now also allowed for the monte carlo integration.
// For those features, the variance "var" can be set to zero
// and a monte carlo approximation for the discrete sum is
// returned.
double KernelDensity::p(const Array< double >& pat) const
{
	double		pp				= 0;
	double		negvar2			= -var * 2;
	unsigned	noSqrDistZero	= 0;
	double		sqrDist			= 0.;


	for (unsigned i = 0; i < size(); ++i) {
		sqrDist = ::sqrDistance(pat, m[ i ]);

		if (var != 0.) {
			pp += exp(sqrDist / negvar2);
		}
		else if (sqrDist == 0.) {
			++noSqrDistZero;
		}
	}
	if (var == 0.) {
		pp = noSqrDistZero / size();
	}
	else {
		pp /= Sqrt2PI * sqrt(var) * size();
	}
	return pp;
}

//===========================================================================

void KernelDensity::estimateVariance()
{
	double meanDist = 0;

	for (unsigned i = 1; i < size(); ++i) {
		for (unsigned j = 0; j < i; ++j) {
			meanDist += euclidianDistance(m[ i ], m[ j ]);
		}
	}
	meanDist /= (size() - 1) * size() * 3;     // 3 = rule of thumb
	var = meanDist * meanDist;
}

//===========================================================================

