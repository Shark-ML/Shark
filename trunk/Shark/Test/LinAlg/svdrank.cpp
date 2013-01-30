#include "shark/LinAlg/svd.h"

#define BOOST_TEST_MODULE LinAlg_svdrank
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

const size_t NumTests=1;
const size_t Dim=2;


double inputMatrix[NumTests][Dim][Dim] =
{
	{
		{1.,  3.},
		{-4.,  3.}
	}
};
//sorted svd vectors of inputMatrix
double svdW[NumTests][Dim]=
{
	{5.14916,2.91309}
};
double svdU[NumTests][Dim][Dim]=
{
	{
		{-0.289784,-0.957092},
		{-0.957092,0.289784}
	}
};
double svdV[NumTests][Dim][Dim]=
{
	{
		{0.687215,-0.726454},
		{-0.726454,-0.687215}
	}
};

//result
double resultRank[NumTests]={2};


BOOST_AUTO_TEST_CASE( LinAlg_svdrank )
{
	for(size_t test=0;test!=NumTests;++test)
	{
		RealMatrix A(2, 2);   // input matrix
		RealMatrix U(2, 2);   // the two orthogonal
		RealMatrix V(2, 2);   // matrices
		RealVector w(2);      // vector for singular values

		// Initializing matrices and vector:
		for (size_t row = 0; row != Dim; row++)
		{
			for (size_t col = 0; col != Dim; col++)
			{
				A(row,col)=inputMatrix[test][row][col];
				U(row, col) = svdU[test][row][col];
				V(row, col) = svdV[test][row][col];
			}
			w(row) = svdW[test][row];
		}

		// Determining rank of matrix A:
		unsigned r = svdrank(A, U, V, w);

		BOOST_CHECK_EQUAL(r,resultRank[test]);
	}
}
