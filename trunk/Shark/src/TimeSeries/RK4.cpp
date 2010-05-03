//===========================================================================
/*!
 *  \file RK4.cpp
 *
 *
 *  \author  Martin Kreutz
 *  \date    21.09.1998
 *
 *  \par Copyright (c) 1998-2003:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      TestData
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of TestData. This library is free software;
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


#ifdef __GNUC__
#pragma implementation
#endif

#include <TimeSeries/RK4.h>

RK4::RK4(unsigned dim, double stepsize)
		: h(stepsize),
		t(0),
		y(dim),
		dy(dim),
		dym(dim),
		dyt(dim),
		yt(dim)
{
	reset();
}

void RK4::reset()
{
	t = 0;
	y = 1.;
}

Array< double > RK4::operator()()
{
	unsigned i;

	derivative(y, dy);

	for (i = 0; i < y.nelem(); ++i)
		yt(i) = y(i) + h * dy(i) / 2;

	derivative(yt, dyt);

	for (i = 0; i < y.nelem(); ++i)
		yt(i) = y(i) + h * dyt(i) / 2;

	derivative(yt, dym);

	for (i = 0; i < y.nelem(); ++i) {
		yt(i) = y(i) + h * dym(i);
		dym(i) += dyt(i);
	}

	derivative(yt, dyt);

	for (i = 0; i < y.nelem(); ++i)
		y(i) = y(i) + h * (dy(i) + dyt(i) + 2 * dym(i)) / 6;

	++t;
	return y;
}

