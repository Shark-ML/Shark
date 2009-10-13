//===========================================================================
/*!
 *  \file eigensort_test.cpp
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

#include <iostream>
#include "Array/ArrayIo.h"
#include "LinAlg/LinAlg.h"

using namespace std;

int main()
{
	Array2D< double > x(3, 3);   // matrix of eigenvectors
	Array  < double > lambda(3); // vector of unsorted eigenvalues
	unsigned          curr_row,  // currently considered matrix
	curr_col;  // row/column

	// Initialization values for matrix of eigenvectors:
	double eigenvectors[9] =
		{
			0.333333,  0.666667, -0.666667,
			0.666667, -0.666667, -0.333333,
			0.666667,  0.333333,  0.666667
		};

	// Initialization values for vector of eigenvalues:
	double eigenvalues[3] =
		{
			3., 9., 6.
		};

	// Initializing eigenvector matrix and eigenvalue vector:
	for (curr_row = 0; curr_row < 3; curr_row++) {
		for (curr_col = 0; curr_col < 3; curr_col++) {
			x(curr_row, curr_col) = eigenvectors[curr_row*3+curr_col];
		}
		lambda(curr_row) = eigenvalues[curr_row];
	}

	// Output before sorting:
	cout << "Output before sorting:\n" << endl;
	cout << "matrix of eigenvectors:" << endl;
	writeArray(x, cout);
	cout << "vector of eigenvalues:" << endl;
	writeArray(lambda, cout);


	// Sorting eigenvectors and eigenvalues:
	eigensort(x, lambda);

	// Output after sorting:
	cout << "\n\nOutput after sorting:\n" << endl;
	cout << "matrix of eigenvectors:" << endl;
	writeArray(x, cout);
	cout << "vector of eigenvalues:" << endl;
	writeArray(lambda, cout);

	// lines below are for self-testing this example, please ignore
	if(lambda(0)==9) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}

