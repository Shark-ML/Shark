#include "shark/LinAlg/svd.h"

#define BOOST_TEST_MODULE LinAlg_svdsort
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

const size_t NumTests=1;
const size_t Dim=3;

double svdW[NumTests][Dim]=
{
	{3,9,6}
};
double svdU[NumTests][Dim][Dim]=
{
	{
		{0.707,  0.707, 0},
		{0.707, -0.707, 0},
		{0,      0,     1}
	}
};
double svdV[NumTests][Dim][Dim]=
{
	{
		{-0.957092, -0.289784, 0},
		{ 0.289784,  0.957092, 0},
		{0,      0,            1}
	}
};

double resultSVDSortW[NumTests][Dim]=
{
	{9,6,3}
};
double resultSVDSortU[NumTests][Dim][Dim]=
{
	{
		{ 0.707, 0, 0.707},
		{-0.707, 0, 0.707},
		{ 0,     1, 0}
	}
};
double resultSVDSortV[NumTests][Dim][Dim]=
{
	{
		{ -0.289784, 0, -0.957092},
		{  0.957092, 0,  0.289784},
		{  0,        1,  0}
	}
};

BOOST_AUTO_TEST_CASE( LinAlg_svd )
{
	for(size_t test=0;test!=NumTests;++test)
	{
		RealMatrix U(Dim,Dim);
		RealMatrix V(Dim,Dim);
		RealVector w(Dim);

		for (size_t row = 0; row != Dim; row++)
		{
			for (size_t col = 0; col != Dim; col++)
			{
				U(row, col) = svdU[test][row][col];
				V(row, col) = svdV[test][row][col];
			}
			w(row) = svdW[test][row];
		}
		//sorting
		svdsort(U,V,w);


		//check w-vector
		for(size_t i=0;i!=Dim;++i)
		{
			BOOST_CHECK_SMALL(w(i)-resultSVDSortW[test][i],1.e-5);
		}
		//check U/V-vector
		for (size_t row = 0; row != Dim; row++)
		{
			for (size_t col = 0; col != Dim; col++)
			{
				BOOST_CHECK_SMALL(U(row , col)-resultSVDSortU[test][row][col],1.e-5);
				BOOST_CHECK_SMALL(V(row , col)-resultSVDSortV[test][row][col],1.e-5);
			}
		}
	}
}
