#include "shark/LinAlg/eigenvalues.h"

#define BOOST_TEST_MODULE LinAlg_detsymm
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
using namespace shark;

BOOST_AUTO_TEST_CASE( LinAlg_detsymm )
{
	const size_t NumTests=5;
	const size_t NumRows=3;
	const size_t NumCols=3;

	double matrices[NumTests][NumRows][NumCols]=
	{
		//Matrix with rank 3 which will be assigned rank 2 due to numerical considerations
		{
			{ 7.,  0.,  0.},
			{-2.,  6.,  0.},
			{ 0., -2.,  0.}
		},
		//Matrix with rank 0
		{
			{ 0, 0, 0.},
			{ 0, 0, 0.},
			{ 0, 0, 0.}
		},
		//Matrix with rank 1
		{
			{ 1, 1, 0.},
			{ 1, 1, 0.},
			{ 0, 0, 0.}
		},
		//Matrix with rank 2
		{
			{ 1, 1, 0.},
			{ 1, 1, 0.},
			{ 0, 0, 4.}
		},
		//Matrix with rank 3
		{
			{ 5, -3, 0.},
			{ -3, 5, 0.},
			{ 0, 0, 4.}
		}
	};

	int results[NumTests]={-28,0,0,0,64};

	for(size_t test = 0; test != NumTests; ++test)
	{
		RealMatrix A(3, 3);   // input matrix
		RealMatrix x(3, 3);   // matrix for eigenvectors
		RealVector lambda(3); // vector for eigenvalues

		// Initializing matrices and vector:
		for (size_t row = 0; row != NumRows; row++) {
			for (size_t col = 0; col < NumCols; col++)
			{
				A(row, col) = matrices[test][row][col];
				x(row, col) = 0.;
			}
			lambda(row) = 0.;
		}

		// Calculating the determinate, eigenvectors and eigenvalues of the matrix:
		double det = detsymm(A, x, lambda);
		double logdet = logdetsymm(A,x,lambda);

		BOOST_CHECK_SMALL( det-results[test],1.e-6 );
		if(test == 4) //only full rank
			BOOST_CHECK_SMALL( logdet-log(double(results[test])), 1.e-6 );
	}
}
