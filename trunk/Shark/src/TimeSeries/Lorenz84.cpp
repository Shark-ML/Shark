//===========================================================================
/*!
 *  \file Lorenz84.cpp
 *
 *
 *  \author  Martin Kreutz
 *  \date    21.09.1998
 *
 *  \par Copyright (c) 1998,2003:
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

#include <SharkDefs.h>
#include <TimeSeries/Lorenz84.h>


Lorenz84::Lorenz84(double apar,
				   double bpar,
				   double Fpar,
				   double Gpar,
				   double steppar)
		: RK4(3, steppar),
		a(apar),
		b(bpar),
		F(Fpar),
		G(Gpar)
{
	reset();
}

void Lorenz84::derivative(const Array< double >& y, Array< double >&dy) const
{
	dy(0) = -Shark::sqr(y(1)) - Shark::sqr(y(2)) - a * y(0) + a * F;
	dy(1) = y(0) * y(1) - b * y(0) * y(2) - y(1) + G;
	dy(2) = b * y(0) * y(1) + y(0) * y(2) - y(2);
}

Generator< Array< double > > * Lorenz84::clone() const
{
	return new Lorenz84(*this);
}

