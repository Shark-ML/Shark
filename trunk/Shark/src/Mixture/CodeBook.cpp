//===========================================================================
/*!
 *  \file CodeBook.cpp
 *
 *  \brief Class for storing and generation code book vectors.
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
 *      <BR>

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
#include <Array/ArrayOp.h>
#include <Rng/GlobalRng.h>
#include <Mixture/CodeBook.h>

//===========================================================================

CodeBook::CodeBook
(
	unsigned numA,
	unsigned dimA
)
		: m(numA, dimA)
{
	//
	// initialize all means with uniformly distributed pseudo random numbers
	//
	for (unsigned i = m.nelem(); i--; m.elem(i) = Rng::uni());
}

//===========================================================================

void CodeBook::resize
(
	unsigned numA,
	bool     copyA
)
{
	m.resize(numA, dim(), copyA);
}

void CodeBook::resize
(
	unsigned numA,
	unsigned dimA,
	bool     copyA
)
{
	m.resize(numA, dimA, copyA);
}

//===========================================================================

unsigned CodeBook::nearest
(
	const Array< double >& v
) const
{
	unsigned i, j;
	double   d, e = 0;

	//
	// compute distances to reference vectors
	//
	for (i = j = 0; i < size(); ++i) {
		//
		// get distance || v - w ||^2
		//
		d = sqrDistance(m[ i ], v);

		//
		// get nearest vector
		//
		if (i == 0 || d < e) {
			j = i;
			e = d;
		}
	}

	return j;
}

//===========================================================================

void CodeBook::nearest
(
	const Array< double >& x,
	Array< double >& y
) const
{
	y = m[ nearest(x)];
}

//===========================================================================

void CodeBook::initialize
(
	const Array< double >& v
)
{
	SIZE_CHECK(v.ndim() == 2)

	Array< double > min(v[ 0 ]);
	Array< double > max(v[ 0 ]);

	for (unsigned i = 1; i < v.dim(0); ++i) {
		for (unsigned j = 0; j < min.dim(0); ++j) {
			if (v(i, j) < min(j)) {
				min(j) = v(i, j);
			}
			if (v(i, j) > max(j)) {
				max(j) = v(i, j);
			}
		}
	}

	initialize(min, max);
}

//===========================================================================

void CodeBook::initialize
(
	const Array< double >& min,
	const Array< double >& max
)
{
	SIZE_CHECK(min.ndim() == 1 && min.samedim(max))

	m.resize(size(), min.dim(0), false);

	for (unsigned i = 0; i < size(); ++i) {
		for (unsigned j = 0; j < dim(); ++j) {
			m(i, j) = Rng::uni(min(j), max(j));
		}
	}
}

//===========================================================================

void CodeBook::kmc
(
	const Array< double >& x,
	double prec,
	unsigned maxiter
)
{
	SIZE_CHECK(x.ndim() == 2 && x.dim(1) == dim() && x.nelem() > 0)

	bool first = true;
	unsigned i, k;
	double   dist;
	Array< double   > mean(size(), dim());
	Array< unsigned > num(size());

	do {
		mean = 0.;
		num  = 0;

		for (k = x.dim(0); k--;) {
			i = nearest(x[ k ]);
			mean[ i ] += x[ k ];
			num(i) ++;
		}

		for (i = size(); i--;) {
			if (num(i)) {
				mean[ i ] /= double(num(i));
			}
			else {
				m[ i ] = mean[ i ] = x[ unsigned(Rng::uni(0, x.dim(0)))];
			}
		}

		if (first) {
			first = false;
			dist  = 10 * prec + 1;
		}
		else {
			Array< double >::iterator oldMeanL = m.begin();
			Array< double >::iterator newMeanL = mean.begin();

			dist = 0;
			while (oldMeanL != m.end()) {
				double dL = fabs(*oldMeanL++ - *newMeanL++);

				if (dist < dL) {
					dist = dL;
				}
			}
		}

		m = mean;
	}
	while (dist > prec && --maxiter != 0);
}

//===========================================================================

