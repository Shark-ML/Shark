#include "shark/LinAlg/eigenvalues.h"

#define BOOST_TEST_MODULE LinAlg_eigenerr
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
using namespace shark;

const size_t Dimensions=3;
double inputMatrix[Dimensions][Dimensions]=
{
	{ 7, -2,  0},
	{-2,  6, -2},
	{ 0, -2,  0}
};
//results based on output of the old shark version
double eigenvectors[Dimensions][Dimensions]=
{
	{ 0.7447626, -0.6624196,  0.0808008},
	{-0.6505405, -0.6936954, 0.3091659},
	{ 0.1487464, 0.2828194,  0.9475693}
};
double eigenvalues[Dimensions]={8.74697419,4.9055710,-0.6525452};



BOOST_AUTO_TEST_CASE( LinAlg_eigenerr )
{
	RealMatrix A(Dimensions, Dimensions);   // input matrix
	RealMatrix x(Dimensions, Dimensions);   // matrix for eigenvectors
	RealVector lambda(Dimensions); // vector for eigenvalues

	// Initializing matrices and vector:
	for (size_t row = 0; row < Dimensions; row++)
	{
		for (size_t col = 0; col < Dimensions; col++)
		{
			A(row, col) = inputMatrix[row][col];
			x(row, col) = eigenvectors[row][col];
		}
		lambda(row) = eigenvalues[row];
	}
	// Calculating relative errors for all eigenvalues:
	for (size_t row = 0; row < Dimensions; row++)
	{
		double error = eigenerr(A, x, lambda, row);
		BOOST_CHECK_SMALL(error,1.e-6);
	}
}










