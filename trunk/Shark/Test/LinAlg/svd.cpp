#include "shark/LinAlg/svd.h"
#include "shark/LinAlg/rotations.h"
#define BOOST_TEST_MODULE LinAlg_svd
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

const size_t NumTests=1;
const size_t Dim=2;

//Testcases
double inputMatrix[NumTests][Dim][Dim] =
{
	{
		{1.,  3.},
		{-4.,  3.}
	}
};
//Testresults
double resultSVD_W[NumTests][Dim]=
{
	{2.91309,5.14916}
};
double resultSVD_U[NumTests][Dim][Dim]=
{
	{
		{-0.957092,-0.289784},
		{0.289784,-0.957092}
	}
};
double resultSVD_V[NumTests][Dim][Dim]=
{
	{
		{-0.726454,0.687215},
		{-0.687215,-0.726454}
	}
};

BOOST_AUTO_TEST_CASE( LinAlg_svd )
{
	for(size_t test=0;test!=NumTests;++test)
	{
		RealMatrix A(Dim,Dim);
		RealMatrix U(Dim,Dim);
		RealMatrix V(Dim,Dim);
		RealVector w(Dim);

		for (size_t row = 0; row != Dim; row++)
		{
			for (size_t col = 0; col != Dim; col++)
			{
				A(row, col) = inputMatrix[test][row][col];
				U(row, col) = 0.;
				V(row, col) = 0.;
			}
			w(row) = 0.;
		}
		//storing svd results
		svd(A, U, V, w);

		//check w-vector
		for(size_t i=0;i!=Dim;++i)
		{
			BOOST_CHECK_SMALL(w(i)-resultSVD_W[test][i],1.e-5);
		}
		//check U/V-vector
		for (size_t row = 0; row != Dim; row++)
		{
			for (size_t col = 0; col != Dim; col++)
			{
				BOOST_CHECK_SMALL(U(row , col)-resultSVD_U[test][row][col],1.e-5);
				BOOST_CHECK_SMALL(V(row , col)-resultSVD_V[test][row][col],1.e-5);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_svd_big )
{
	for(std::size_t test = 0; test != 10; ++test){
		//first generate a suitable eigenvalue problem matrix A
		RealMatrix R = blas::randomRotationMatrix(5);
		RealMatrix lambda(5,5);
		lambda.clear();
		for(std::size_t i = 0; i != 3; ++i){
			lambda(i,i) = i;
		}
		RealMatrix A = prod(R,lambda);
		A = prod(A,trans(R));
		
		RealMatrix U(5,5);
		RealMatrix V(5,5);
		RealVector w(5);

		
		//storing svd results
		svd(A, U, V, w);
		
		//recreate A
		for(std::size_t i = 0; i != 5; ++i){
			lambda(i,i) = w(i);
		}
		
		RealMatrix ATest = prod(U,lambda);
		ATest = prod(ATest,trans(V));
		//test reconstruction error
		BOOST_CHECK_SMALL(norm_1(A-ATest)/25.0,1.e-12);
	}
}


