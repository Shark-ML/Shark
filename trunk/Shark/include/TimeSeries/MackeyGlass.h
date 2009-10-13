//===========================================================================
/*!
 *  \file MackeyGlass.h
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



#ifndef __cplusplus
#error Must use C++.
#endif

#ifndef __MACKEYGLASS_H
#define __MACKEYGLASS_H

#ifdef __GNUC__
#pragma interface
#endif

#include <iostream>
#include <TimeSeries/RK4-1D.h>

//===========================================================================

class MackeyGlass : public RK4_1D
{
public:
	MackeyGlass(double   a    = 0.2,
				double   b    = 10,
				double   c    = 0.1,
				unsigned tau  = 30,
				double   step = 1,
				unsigned subsample = 1,
				double   initial   = 0.9);

	void   reset();
	double operator()();

private:
	double   A, B, C;
	unsigned T;
	unsigned subsample;
	double   initial;
	Array< double > x;

	double derivative(double) const;

	Generator< double >* clone() const;
};

//===========================================================================

#endif /* !__MACKEYGLASS_H */

