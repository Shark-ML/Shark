#include <shark/LinAlg/Inverse.h>

#define BOOST_TEST_MODULE LinAlg_invertSymmPositiveDefinite
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

const size_t Dimensions=4;

double inputMatrix[Dimensions][Dimensions]=
{
	{ 9,   3, -6,  12},
	{ 3,  26, -7, -11},
	{-6, -7,   9,   7},
	{12, -11,  7,  65}
};

double invertedMatrix[Dimensions][Dimensions]=
{
	{  2.98333,  0.01667,  2.65, -0.83333},
    {  0.01667,  0.05   ,  0.05,  0},
    {  2.65   ,  0.05   ,  2.5 , -0.75},
    { -0.83333,  0      , -0.75,  0.25}
};


BOOST_AUTO_TEST_CASE( LinAlg_invertSymmPositiveDefinite )
{
	RealMatrix C(Dimensions, Dimensions);   // input matrix
	RealMatrix inverse(Dimensions, Dimensions);   // inverted matrix

	// Initializing matrices
	for (size_t row = 0; row < Dimensions; row++)
	{
		for (size_t col = 0; col < Dimensions; col++)
		{
			C(row, col) = inputMatrix[row][col];
			inverse(row, col) = 0;
		}
	}
	//invert
	invertSymmPositiveDefinite(inverse,C);

	std::cout<<inverse<<std::endl;

	//test for equality
	for (size_t row = 0; row < Dimensions; row++)
	{
		for (size_t col = 0; col < Dimensions; col++)
		{
			BOOST_CHECK_SMALL(inverse(row, col)-invertedMatrix[row][col],1.e-5);
		}
	}
}
