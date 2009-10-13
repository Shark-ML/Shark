//===========================================================================
/*!
 *  \file LocalLinearRegression.cpp
 *
 *  \brief Localized linear regression model
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
#include <Array/ArrayOp.h>
#include <Mixture/LocalLinearRegression.h>

//===========================================================================

LocalLinearRegression::LocalLinearRegression(unsigned m)
		: CodeBook(m, 0), lr(m)
{}

//===========================================================================

void LocalLinearRegression::reset()
{
	for (unsigned i = 0; i < size(); ++i) {
		lr(i).reset();
	}
}

//===========================================================================

void LocalLinearRegression::update()
{
	for (unsigned i = 0; i < size(); ++i) {
		lr(i).update();
	}
}

//===========================================================================

void LocalLinearRegression::train
(
	const Array< double >& x,
	const Array< double >& y
)
{
	SIZE_CHECK(x.ndim() > 0 && x.ndim() <= 2)

	unsigned index;

	if (x.ndim() == 1) {
		index = nearest(x);
		lr(index).train(x - m[ index ], y);
	}
	else {
		SIZE_CHECK(x.dim(0) == y.dim(0))

		for (unsigned i = 0, j = 0; i < x.dim(0); ++i, j += x.dim(1)) {
			index = nearest(x[ i ]);
			lr(index).train(x[ i ] - m[ index ], y[ i ]);
		}
	}
}

//===========================================================================

void LocalLinearRegression::recall
(
	const Array< double >& x,
	Array< double >& y
)
{
	SIZE_CHECK(x.ndim() > 0 && x.ndim() <= 2)

	unsigned index;

	if (x.ndim() == 1) {
		index = nearest(x);
		lr(index).recall(x - m[ index ], y);
	}
	else {
		y.resize(x.dim(0), lr(0).syM.dim(0), false);

		for (unsigned i = 0, j = 0; i < x.dim(0); ++i, j += x.dim(1)) {
			ArrayReference< double > yi(y[ i ]);
			index = nearest(x[ i ]);
			lr(index).recall(x[ i ] - m[ index ], yi);
		}
	}
}

//===========================================================================

void LocalLinearRegression::resize
(
	unsigned numA,
	bool     copyA
)
{
	CodeBook::resize(numA, copyA);
	lr.resize(numA, copyA);
}

void LocalLinearRegression::resize
(
	unsigned numA,
	unsigned dimA,
	bool     copyA
)
{
	CodeBook::resize(numA, dimA, copyA);
	lr.resize(numA, copyA);
}

//===========================================================================

