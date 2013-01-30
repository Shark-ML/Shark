#include "shark/LinAlg/VectorStatistics.h"

#define BOOST_TEST_MODULE LinAlg_covar_corrcoef
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
using namespace shark;

const unsigned max_rows = 2;
const unsigned max_cols = 3;
double init_values[ max_rows ][ max_cols ] =
{
	{1., 1., 3.},
	{ -4., 3., 2.}
};

double resultCovMat[max_cols][max_cols]=
{
	{6.25,-2.5,1.25},
	{-2.5, 1,  -0.5},
	{1.25,-0.5,0.25}
};

double resultCorrCoeff[max_cols][max_cols]=
{
	{1, -1, 1},
	{-1, 1,-1},
	{1, -1, 1}
};


BOOST_AUTO_TEST_CASE( LinAlg_covar_corrcoef )
{

	// The data vector matrix, the two single data vectors,
	// the covariance and coefficient of correlation matrices
	// for the data vector matrix:
	std::vector<RealVector> dataVec(max_rows, RealVector(max_cols));

	// Initialize data vector matrix and single data vectors
	// with the same values:
	for (size_t row = 0; row < max_rows; row++) {
		for (size_t col = 0; col < max_cols; col++) {
			dataVec[row](col) = init_values[ row ][ col ];
		}
	}
	Data<RealVector> data(dataVec);
	RealMatrix covar_mat = covariance(data);

	RealMatrix corrcoef_mat = corrcoef(data);

	for (size_t row = 0; row < max_rows; row++)
	{
		for (size_t col = 0; col < max_cols; col++)
		{
			BOOST_CHECK_SMALL(covar_mat(row,col)-resultCovMat[row][col],1.e-14);
			BOOST_CHECK_SMALL(corrcoef_mat(row,col)-resultCorrCoeff[row][col],1.e-14);
		}
	}
}
