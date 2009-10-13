//===========================================================================
/*!
 *  \file fft_test.cpp
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
 *      LinAlg
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of LinAlg. This library is free software;
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

#include "Array/ArrayIo.h"
#include "LinAlg/fft.h"

using namespace std;


// Simple test function which is sampled:
double func(double t)
{
	return t * t;
}


int main()
{
	int                             t;     // Time.
	complex< double >               ft;    // Function value f( t ).
	Array< std::complex< double > > data;  // f( t ) for all sample points t.


	// Create and save sample data (as real value < ft, 0 >):
	for (t = -8; t < 8; t++) {
		ft = func((double)t);
		data.append_elem(ft);
	}

	// Output of original data (in time domain):
	cout << "Original data (in time domain):" << endl;
	writeArray(data, cout);
	cout << endl;

	// Map function from time domain to frequency domain:
	fft(data);
	cout << "Data in frequency domain:" << endl;
	writeArray(data, cout);
	cout << endl;

	// Map function from frequency domain to time domain
	// (inverse fourier transformation):
	ifft(data);
	cout << "Data in time domain (after inverse fourier transformation):"
	<< endl;
	writeArray(data, cout);

	// lines below are for self-testing this example, please ignore
	if (fabs(data(0).real() - 64.0) + fabs(data(0).imag()) <= 1.e-15) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
