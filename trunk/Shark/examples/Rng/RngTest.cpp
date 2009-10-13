//===========================================================================
/*!
 *  \file RngTest.cpp
 *
 *
 *  \author  Martin Kreutz
 *  \date    21.08.1998
 *
 *  \par Copyright (c) 1999,2003:
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
 *      Rng
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Rng. This library is free software;
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


#include <iostream>
#include <vector>
#include "Rng/DiffGeometric.h"

using namespace std;

int main()
{
	const long   NumSamples = 1000000;
	const long   HistoBins  = 21;
	const double P          = 0.3;

	long i, z;
	vector< double > histo(HistoBins);
	Geometric     geom(P);
	DiffGeometric dgeom(P);

	for (i = 0; i < HistoBins; ++i)
		histo[ i ] = 0;

	for (i = 0; i < NumSamples; ++i) {
		z = geom();
		if (z >= 0 && z < HistoBins)
			histo[ z ]++;
	}

	for (i = 0; i < HistoBins; ++i)
		cout << i << ":\t"
		<< histo[ i ] / NumSamples << '\t'
		<< geom.p(i) << endl;

	cout << endl;

	for (i = 0; i < HistoBins; ++i)
		histo[ i ] = 0;

	for (i = 0; i < NumSamples; ++i) {
		z = dgeom();
		if (z >= -HistoBins / 2 && z <= HistoBins / 2)
			histo[ z+HistoBins/2 ]++;
	}

	for (i = 0; i < HistoBins; ++i)
		cout << i << ":\t"
		<< histo[ i ] / NumSamples << '\t'
		<< dgeom.p(i - HistoBins / 2) << endl;

	return 0;
}

