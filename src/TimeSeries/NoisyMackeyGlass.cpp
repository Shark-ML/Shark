//===========================================================================
/*!
 *  \file NoisyMackeyGlass.cpp
 *
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

#include <TimeSeries/NoisyMackeyGlass.h>

NoisyMackeyGlass::NoisyMackeyGlass(double   a,
								   double   b,
								   double   c,
								   unsigned tau,
								   double   sigma,
								   bool     dyn)
		: A(a),
		B(b),
		C(c),
		T(tau),
		t(0),
		variance(sigma * sigma),
		dynamic(dyn),
		x(tau)
{
	reset();
}

void NoisyMackeyGlass::reset()
{
	t = T;
	x = 0.9;
}

double NoisyMackeyGlass::operator()()
{
	double xT = x(t % T);
	double z  = variance > 0 ? gauss(0, variance) : 0;

	x(t % T) = A * xT / (1 + pow(xT, B)) + (1 - C) * x((t - 1) % T);

	return dynamic ? x(t++ % T) += z : x(t++ % T) + z;
}

Generator< double >* NoisyMackeyGlass::clone() const
{
	return new NoisyMackeyGlass(*this);
}

