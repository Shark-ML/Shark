//===========================================================================
/*!
 *  \file g_inverse_matrix.cpp
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

#include<iostream>
#include "Array/ArrayIo.h"
#include "LinAlg/LinAlg.h"

using namespace std;


// Values used for initialization of input matrix:
double values[ 9 ] = {
						 1., 2., -1.,
						 2., 5., -1.,
						 1., 2.,  0.

					 };


int main()
{
	unsigned          i, j;      // Counter variables.
	Array2D< double > A(3, 3),   // Input matrix.
	B(3, 3);   // Generalized inverse matrix.


	// Initialize matrices:
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			A(i, j) = values[ i * 3 + j ];
		}
	}
	B = .0;

	// Output of input matrix:
	cout << "input matrix:" << endl;
	writeArray(A, cout);

	// Calculate generalized inverse matrix:
	g_inverse(A, B);

	// Output of generalized inverse matrix:
	cout << "generalized inverse matrix:" << endl;
	writeArray(B, cout);

	// lines below are for self-testing this example, please ignore
	if(B(2,1)<=1.e-15) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}

