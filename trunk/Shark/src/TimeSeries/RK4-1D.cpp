//===========================================================================
/*!
 *  \file RK4-1D.cpp
 *
 *
 *  \author  Martin Kreutz
 *  \date    15.10.1998
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

#include <TimeSeries/RK4-1D.h>

RK4_1D::RK4_1D(double stepsize)
		: h(stepsize), t(0), y(1)
{}

void RK4_1D::reset()
{
	t = 0;
	y = 1.;
}

double RK4_1D::operator()()
{
	dy = derivative(y);

	yt = y + h * dy / 2;

	dyt = derivative(yt);

	yt = y + h * dyt / 2;

	dym = derivative(yt);

	yt = y + h * dym;
	dym += dyt;

	dyt = derivative(yt);

	y = y + h * (dy + dyt + 2 * dym) / 6;

	++t;
	return y;
}

/*
double RK4_1D::operator ( ) ( )
{
    dym = derivative( y + h * derivative( y + h * derivative( y ) / 2 ) / 2 );

    yt = y + h * dym;
    dym += dyt;

    dyt = derivative( yt );

    y = y + h * ( dy + dyt + 2 * dym ) / 6;

    ++t;
    return y;
}
*/

