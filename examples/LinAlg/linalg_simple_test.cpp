//===========================================================================
/*!
 *  \file LinAlg_simple_test.cpp
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

#include "Array/Array.h"
#include "Array/ArrayIo.h"
#include "Rng/GlobalRng.h"
#include "LinAlg/LinAlg.h"
#include "SharkDefs.h"

using namespace std;

// Will initialize a vector or 2-dimensional matrix
// by random numbers:
//
void init_array(Array< double > &a)
{
	Uniform  uni;           // Random number generator:
	unsigned no_dims(0),    // No. of Array dimensions.
	no_rows(0),    // No. of rows (matrix) or columns (vector).
	no_cols(0),    // No. of columns (matrix).
	curr_row,      // Current row (matrix) or column (vector).
	curr_col;      // Current column (matrix).


	// Determine no. of dimensions, rows and columns:
	no_dims = a.ndim();
	if (no_dims < 1 || no_dims > 2) {
	  throw SHARKEXCEPTION("Can't initialize array - dimension error!");
	}
	no_rows = a.dim(0);
	if (no_dims > 1) no_cols = a.dim(1);

	// Initialize Array:
	for (curr_row = 0; curr_row < no_rows; curr_row++) {
		// Vector:
		if (no_dims == 1) {
			a(curr_row) = uni(-9., 9.);
		}
		else {
			// Matrix:
			for (curr_col = 0; curr_col < no_cols; curr_col++) {
				a(curr_row, curr_col) = uni(-9., 9.);
			}
		}
	}

	return;
}


// Example for function "mean(...)":
bool test_mean(Array< double > &input)
{
	Array< double > mean_vec; // Vector of mean values.

	// Clear result vector:
	mean_vec = 0.;
	// Calculate mean vector:
	mean_vec = mean(input);
	// Output of input matrix and mean values vector:
	cout << "input matrix:" << endl;
	writeArray(input, cout);
	cout << "vector of matrix mean values:" << endl;
	writeArray(mean_vec, cout);
	cout << endl << endl;

	// lines below are for self-testing this example, please ignore
	if(fabs(mean_vec(0)-1.51684)<1.e-5) return true;
	else return false;
}


// Example for function "variance(...)":
bool test_variance(Array< double > &input)
{
	Array< double > var_vec; // Vector of variance values.

	// Clear result vector:
	var_vec = 0.;
	// Calculate variance vector:
	var_vec = variance(input);
	// Output of input matrix and variance vector:
	cout << "input matrix:" << endl;
	writeArray(input, cout);
	cout << "vector of matrix variance values:" << endl;
	writeArray(var_vec, cout);
	cout << endl << endl;

	// lines below are for self-testing this example, please ignore
	if(fabs(var_vec(0)-8.78551)<1.e-5) return true;
	else return false;
}

// Example for function "angle(...)":
bool test_angle(Array< double > &copy_from_matrix)
{
	Array< double > x(4),       // Input vectors.
	y(4);
	double          angl(0.);   // Angle between vectors.


	// Initialize vectors by copying rows no. 1 and 2
	// of global input matrix:
	x = copy_from_matrix[ 0 ];
	y = copy_from_matrix[ 1 ];


	// Calculate angle between the input vectors:
	angl = angle(x, y);
	// Output of input vectors and angle value:
	cout << "input vector x:" << endl;
	writeArray(x, cout);
	cout << endl;
	cout << "input vector y:" << endl;
	writeArray(y, cout);
	cout << endl;
	cout << "angle = " << angl << endl;
	cout << endl << endl;

	// lines below are for self-testing this example, please ignore
	if(fabs(angl- -0.362842)<1.e-5) return true;
	else return false;
}


// Example for function "transpose(...)":
bool test_transpose(Array< double > &input)
{
	Array< double > trans_matrix; // Transposed matrix.

	// Clear result matrix:
	trans_matrix = 0.;
	// Calculate transpose matrix:
	trans_matrix = transpose(input);
	// Output of input matrix and transposed matrix:
	cout << "input matrix:" << endl;
	writeArray(input, cout);
	cout << endl;
	cout << "transposed matrix:" << endl;
	writeArray(trans_matrix, cout);
	cout << endl << endl;

	// lines below are for self-testing this example, please ignore
	if(fabs(trans_matrix(1,0)-1.60387)<1.e-5) return true;
	else return false;
}


// Example for function "diagonal(...)":
bool test_diagonal()
{
	Array< double > input_vec(4);   // Input vector.
	Array< double > matrix;         // Diagonal matrix.


	init_array(input_vec);
	// Clear result matrix:
	matrix = 0.;
	// Create diagonal matrix:
	matrix = diagonal(input_vec);
	// Output of input vector and diagonal matrix:
	cout << "input vector:" << endl;
	writeArray(input_vec, cout);
	cout << endl;
	cout << "diagonal matrix:" << endl;
	writeArray(matrix, cout);
	cout << endl << endl;

	// lines below are for self-testing this example, please ignore
	if(fabs(matrix(0,0)- -2.0725)<1.e-5) return true;
	else return false;
}


// Example for function "trace(...)":
bool test_trace(Array< double > input)
{
	double diag_sum(0.);   // Sum of diagonal values.

	// Determine sum of diagonal values of input matrix:
	diag_sum = trace(input);
	// Output of input matrix and sum of diagonal values:
	cout << "input matrix:" << endl;
	writeArray(input, cout);
	cout << endl;
	cout << "sum of diagonal values of matrix = " << diag_sum << endl;
	cout << endl << endl;

	// lines below are for self-testing this example, please ignore
	if(fabs(diag_sum- -3.64975)<1.e-5) return true;
	else return false;
}

// Example for function "meanvar(...)":
bool test_meanvar(Array< double > input)
{
	Array< double > mean_vec, // Vector of mean values.
	var_vec;  // Variances vector.

	// Clear result vectors:
	mean_vec = 0.;
	var_vec = 0.;

	// Calculate mean and variance values:
	meanvar(input, mean_vec, var_vec);
	// Output of input matrix and result vectors:
	cout << "input matrix:" << endl;
	writeArray(input, cout);
	cout << "\nvector of mean values:" << endl;
	writeArray(mean_vec, cout);
	cout << "\nvariances vector:" << endl;
	writeArray(var_vec, cout);
	cout << endl << endl;

	// lines below are for self-testing this example, please ignore
	if(fabs(var_vec(0)-8.78551)<1.e-5) return true;
	else return false;
}


int main()
{
	Array< double > input_matrix(4, 4);

	//for self-testing, please ignore:
	bool testmean,testvar, testangle,testtrans,testdiag, testtrace, testmeanvar;
	// end self-testing block

	// Initialize global input matrix by random numbers:
	init_array(input_matrix);

	cout << "Testing function \"mean\":" << endl;
	testmean=test_mean(input_matrix);
	cout << "----------------------------------------" << endl;
	cout << "Testing function \"variance\":" << endl;
	testvar=test_variance(input_matrix);
	cout << "----------------------------------------" << endl;
	cout << "Testing function \"angle\":" << endl;
	testangle=test_angle(input_matrix);
	cout << "----------------------------------------" << endl;
	cout << "Testing function \"transpose\":" << endl;
	testtrans=test_transpose(input_matrix);
	cout << "----------------------------------------" << endl;
	cout << "Testing function \"diagonal\":" << endl;
	testdiag=test_diagonal();
	cout << "----------------------------------------" << endl;
	cout << "Testing function \"trace\":" << endl;
	testtrace=test_trace(input_matrix);
	cout << "----------------------------------------" << endl;
	cout << "Testing function \"meanvar\":" << endl;
	testmeanvar=test_meanvar(input_matrix);
	cout << "----------------------------------------" << endl;

	// lines below are for self-testing this example, please ignore
	if (testmean && testvar &&  testangle && testtrans && testdiag &&  testtrace && testmeanvar) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}


