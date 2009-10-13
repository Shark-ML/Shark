//===========================================================================
/*!
 *  \file BimodalBrownianProcess.cpp
 *
 *
 *  \author  Martin Kreutz
 *  \date    25.02.1999
 *
 *  \par Copyright (c) 1999, 2003:
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

#include <TimeSeries/BimodalBrownianProcess.h>

BimodalBrownianProcess::BimodalBrownianProcess(double masspar,
		double alphapar,
		double sigmapar,
		double steppar)
		: RK4(2, steppar),
		mass(masspar),
		alpha(alphapar),
		time(0),
		noise(0),
		gauss(0, sigmapar)
{
	reset();
}

void BimodalBrownianProcess::derivative(const Array< double >& y,
										Array< double >& dy) const
{
	dy(0) = y(1);
	dy(1) = 4 * y(0) * (1 - y(0) * y(0)) / mass
			- alpha * y(1) + noise;
}

Array< double > BimodalBrownianProcess::operator()()
{
	if ((time += h) > 1) {
		time -= 1;
		noise = gauss();
	}
	return RK4::operator()();
}

Generator< Array< double > > * BimodalBrownianProcess::clone() const
{
	return new BimodalBrownianProcess(*this);
}

