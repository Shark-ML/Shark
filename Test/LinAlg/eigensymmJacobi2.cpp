#include "shark/LinAlg/eigenvalues.h"

#define BOOST_TEST_MODULE LinAlg_eigensymmJacobi2
#include <boost/test/unit_test.hpp>

using namespace shark;

BOOST_AUTO_TEST_CASE( LinAlg_eigensymmJacobi2 )
{
	RealMatrix A(3, 3);   // input matrix
	RealMatrix x(3, 3);   // matrix for eigenvectors
	RealVector lambda(3); // vector for eigenvalues

	// Initialization values for input matrix:
	double upper_triangle[9] =
		{
			7., -2.,  0.,

			0.,  6., -2.,

			0.,  0.,  5.
		};

	// Initializing matrices and vector:
	for (size_t curr_row = 0; curr_row < 3; curr_row++) {
		for (size_t curr_col = 0; curr_col < 3; curr_col++) {
			A(curr_row, curr_col) = upper_triangle[curr_row*3+curr_col];
			x(curr_row, curr_col) = 0.;
		}
		lambda(curr_row) = 0.;
	}

	// Calculating eigenvalues and eigenvectors:
	eigensymmJacobi2(A, x, lambda);

	BOOST_CHECK_EQUAL(lambda(0),7);
}
