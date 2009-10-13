//===========================================================================
/*!
 *  \file DiscreteMackeyGlass.cpp
 *
 *  \author  Martin Kreutz
 *  \date    16.09.1998
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

#include <cmath>
#include <TimeSeries/DiscreteMackeyGlass.h>

DiscreteMackeyGlass::DiscreteMackeyGlass(double a,
		double b,
		double c,
		unsigned tau)
		: A(a), B(b), C(c), T(tau), t(0), x(tau)
{
	reset();
}

void DiscreteMackeyGlass::reset()
{
	t = T;
	x = 0.9;
}

double DiscreteMackeyGlass::operator()()
{
	double xT = x(t % T);
	x(t % T) = A * xT / (1 + pow(xT, B)) + (1 - C) * x((t - 1) % T);
	return x(t++ % T);
}

Generator< double >* DiscreteMackeyGlass::clone() const
{
	return new DiscreteMackeyGlass(*this);
}

