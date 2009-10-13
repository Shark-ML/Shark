//===========================================================================
/*!
 *  \file svdrank_test.cpp
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
	Array2D< double > A(2, 2);   // input matrix
	Array2D< double > U(2, 2);   // the two orthogonal
	Array2D< double > V(2, 2);   // matrices
	Array  < double > w(2);      // vector for singular values
	unsigned          curr_row,  // currently considered matrix
	curr_col,  // row/column
	r(0);      // rank of input matrix A

	// Initialization values for input matrix:
	double input_matrix[4] =
		{
			1.,  3.,

			-4.,  3.
		};

	// Initializing matrices and vector:
	for (curr_row = 0; curr_row < 2; curr_row++) {
		for (curr_col = 0; curr_col < 2; curr_col++) {
			A(curr_row, curr_col) = input_matrix[curr_row*2+curr_col];
			U(curr_row, curr_col) = 0.;
			V(curr_row, curr_col) = 0.;
		}
		w(curr_row) = 0.;
	}

	// Singular value decomposition:
	svd(A, U, V, w);

	// Sorting singular values by descending order:
	svdsort(U, V, w);

	// Determining rank of matrix A:
	r = svdrank(A, U, V, w);

	// Output of matrices, singular values and rank:
	cout << "input matrix:" << endl;
	writeArray(A, cout);
	cout << "column-orthogonal matrix U:" << endl;
	writeArray(U, cout);
	cout << "orthogonal matrix V:" << endl;
	writeArray(V, cout);
	cout << "vector of singular values:" << endl;
	writeArray(w, cout);
	cout << "\nrank of matrix A is " << r << endl;

	// lines below are for self-testing this example, please ignore
	if(fabs(w(0)- 5.14916)<1.e-5) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}

