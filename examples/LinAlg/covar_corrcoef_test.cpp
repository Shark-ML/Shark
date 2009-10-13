//===========================================================================
/*!
 *  \file covar_corrcoef_test.cpp
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
#include "LinAlg/LinAlg.h"

using namespace std;

// No. of rows and columns of the
// data vector matrix:
const unsigned max_rows = 2;
const unsigned max_cols = 3;

// Values used for the initialization
// of the data vector matrix and the
// single data vectors:
double init_values[ max_rows ][ max_cols ] =
	{
		{
			1., 1., 3.
		},
		{ -4., 3., 2.}
	};


int main()
{
	// Current row and column of the data vector matrix:
	unsigned        curr_row, curr_col;

	// The data vector matrix, the two single data vectors,
	// the covariance and coefficient of correlation matrices
	// for the data vector matrix:
	Array< double > input_mat(max_rows, max_cols),
	x_vec(max_cols), y_vec(max_cols),
	covar_mat, corrcoef_mat;

	// The covariance and coefficient of correlation values for
	// the single data vectors:
	double          covar, corrcoeff;

	// Initialize data vector matrix and single data vectors
	// with the same values:
	for (curr_row = 0; curr_row < max_rows; curr_row++) {
		for (curr_col = 0; curr_col < max_cols; curr_col++) {
			input_mat(curr_row, curr_col) = init_values[ curr_row ][ curr_col ];
			if (curr_row == 0) {
				x_vec(curr_col) = init_values[ curr_row ][ curr_col ];
			}
			else {
				y_vec(curr_col) = init_values[ curr_row ][ curr_col ];
			}
		}
	}

	// Calculate covariance matrix for the data vector matrix
	// and covariance value for the single data vectors:
	covar_mat = covariance(input_mat);
	covar = covariance(x_vec, y_vec);

	// Calculate coefficient of correlation matrix for the data vector matrix
	// and coefficient of correlation value for the single data vectors:
	corrcoef_mat = corrcoef(input_mat);
	corrcoeff = corrcoef(x_vec, y_vec);

	// Output of the input matrix and vectors and the results:
	cout << "input matrix:" << endl;
	writeArray(input_mat, cout);
	cout << "covariance matrix for the input matrix:" << endl;
	writeArray(covar_mat, cout);
	cout << "coefficient of correlation matrix for the input matrix:" << endl;
	writeArray(corrcoef_mat, cout);

	cout << "data vector x:" << endl;
	writeArray(x_vec, cout);
	cout << "data vector y:" << endl;
	writeArray(y_vec, cout);
	cout << "covariance for the data vectors x and y: " << covar << endl;
	cout << "coefficient of correlation for the data vectors x and y: "
	<< corrcoeff << endl;

	// lines below are for self-testing this example, please ignore
	if(fabs(corrcoeff- 0.381246)<1.e-6) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}





