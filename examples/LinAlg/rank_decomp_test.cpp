//===========================================================================
/*!
 *  \file rank_decomp_test.cpp
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
	Array2D< double > A(3, 3);   // input matrix
	Array2D< double > h(3, 3);   // matrix witrh intermediate results
	Array2D< double > x(3, 3);   // matrix for eigenvectors
	Array  < double > lambda(3); // vector for eigenvalues
	unsigned          curr_row,  // currently considered matrix
	curr_col,  // row/column
	r;         // rank of the matrix

	// Initialization values for input matrix:
	double bottom_triangle[9] =
		{
			7.,  0.,  0.,

			-2.,  6.,  0.,

			0., -2.,  0.
		};

	// Initializing matrices and vector:
	for (curr_row = 0; curr_row < 3; curr_row++) {
		for (curr_col = 0; curr_col < 3; curr_col++) {
			A(curr_row, curr_col) = bottom_triangle[curr_row*3+curr_col];
			x(curr_row, curr_col) = 0.;
		}
		lambda(curr_row) = 0.;
	}

	// Output of input matrix:
	cout << "input matrix (not symmetric):" << endl;
	writeArray(A, cout);

	// Calculating the rank, eigenvectors and eigenvalues of the matrix:
	r = rankDecomp(A, x, h, lambda);

	// Output of input matrix:
	cout << "input matrix after rankDecomp:" << endl;
	writeArray(A, cout);

	// Output of rank:
	cout << "rank of matrix is " << r << endl << endl;

	// Output of eigenvector matrix and eigenvalue vector:
	cout << "matrix of eigenvectors:" << endl;
	writeArray(x, cout);
	cout << "vector of eigenvalues:" << endl;
	writeArray(lambda, cout);

	// lines below are for self-testing this example, please ignore
	if(fabs(lambda(0)- 8.74697)<1.e-5) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}

