//===========================================================================
/*!
 *  \file lorenz84.cpp
 *
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




#include <iostream>
#include <TimeSeries/Lorenz84.h>
#include <TimeSeries/SelectComponent.h>

using namespace std;

int main()
{
	const unsigned Num = 10;

	unsigned i;
	Array< double > xyz;
	Lorenz84 lorenz84;
	SelectComponent< double > x(lorenz84, 0);
	SelectComponent< double > y(lorenz84, 1);
	SelectComponent< double > z(lorenz84, 2);

	for (i = 0; i < Num; ++i) {
		xyz = lorenz84();
		cout << xyz(0) << '\t' << xyz(1) << '\t' << xyz(2) << endl;
	}

	cout << endl;

	for (i = 0; i < Num; ++i)
		cout << x() << '\t' << y() << '\t' << z() << endl;

	return 0;
}


