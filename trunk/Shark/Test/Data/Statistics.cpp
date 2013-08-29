#include "shark/Data/Statistics.h"

#define BOOST_TEST_MODULE Data_Statistics
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

const std::size_t Dimensions=4;
struct StatisticsFixture
{
	UnlabeledData<RealVector> inputData;
	UnlabeledData<RealVector> inputDataSmallBatch;
	StatisticsFixture()
	{
		//values of the input matrix
		double vals[Dimensions][Dimensions]=
		{
			{0.5,    2,   1.3,  2.6},
			{1.7,    2.4, 1.3, -4  },
			{-2,     3.1, 2,    2.3},
			{-0.25, -1,   2.1, -0.8}
		};
		std::vector<RealVector> vec(Dimensions,RealVector(4));
		for(std::size_t row=0;row!= Dimensions;++row)
		{
			for(std::size_t col=0;col!=Dimensions;++col)
			{
				vec[row](col) = vals[row][col];
			}
		}
		inputData = createDataFromRange(vec);
		inputDataSmallBatch = createDataFromRange(vec,1);
	}
};

//results
double resultMean[Dimensions]={-0.0125,1.625,1.675,0.025};
double resultVariance[Dimensions][Dimensions]=
{
	{ 1.80047,-0.19718,-0.395312,-2.47468},
	{-0.19718, 2.45188, -0.26687, 0.84187},
	{-0.39531,-0.26687, 0.141875, 0.23312},
	{-2.47468, 0.84187, 0.233125, 7.17188}
};

BOOST_FIXTURE_TEST_SUITE(data, StatisticsFixture);

BOOST_AUTO_TEST_CASE( Data_Statistics_mean )
{
	// Calculate mean vector:
	RealVector meanVec = mean(inputData);
	RealVector meanVecSmall = mean(inputDataSmallBatch);

	for(std::size_t i=0;i!=Dimensions;++i)
	{
		BOOST_CHECK_SMALL(meanVec(i)-resultMean[i],1.e-5);
		BOOST_CHECK_SMALL(meanVecSmall(i)-resultMean[i],1.e-5);
	}
}


BOOST_AUTO_TEST_CASE( Data_Statistics_variance )
{
	// Calculate variance vector:
	RealVector varVec = variance(inputData);
	RealVector varVecSmall = variance(inputDataSmallBatch);

	for(std::size_t i=0;i!=Dimensions;++i)
	{
		BOOST_CHECK_SMALL(varVec(i)-resultVariance[i][i],1.e-5);
		BOOST_CHECK_SMALL(varVecSmall(i)-resultVariance[i][i],1.e-5);
	}
}

BOOST_AUTO_TEST_CASE( Data_Statistics_meanvar )
{
	RealVector meanVec;
	RealVector varVec;
	
	RealVector meanVecAlternative;
	RealVector varVecAlternative;
	
	RealVector meanVecSmall;
	RealVector varVecSmall;

	// Calculate mean and variance values:
	meanvar(inputData, meanVec, varVec);
	meanvar(inputDataSmallBatch, meanVecSmall, varVecSmall);
	for(std::size_t i=0;i!=Dimensions;++i)
	{
		BOOST_CHECK_SMALL(meanVec(i)-resultMean[i],1.e-5);
		BOOST_CHECK_SMALL(varVec(i)-resultVariance[i][i],1.e-5);
		BOOST_CHECK_SMALL(meanVecSmall(i)-resultMean[i],1.e-5);
		BOOST_CHECK_SMALL(varVecSmall(i)-resultVariance[i][i],1.e-5);
	}
}
BOOST_AUTO_TEST_CASE( Data_Statistics_meanvar_covariance )
{
	RealVector meanVec;
	RealMatrix varMat;
	
	RealVector meanVecSmall;
	RealMatrix varMatSmall;

	// Calculate mean and variance values:
	meanvar(inputData, meanVec, varMat);
	meanvar(inputDataSmallBatch, meanVecSmall, varMatSmall);
	for(std::size_t i=0;i!=Dimensions;++i)
	{
		BOOST_CHECK_SMALL(meanVec(i)-resultMean[i],1.e-5);
		BOOST_CHECK_SMALL(meanVecSmall(i)-resultMean[i],1.e-5);
		for(std::size_t j=0; j != Dimensions; ++j){
			BOOST_CHECK_SMALL(varMat(i,j)-resultVariance[i][j],1.e-5);
			BOOST_CHECK_SMALL(varMatSmall(i,j)-resultVariance[i][j],1.e-5);
		}
	}
}


BOOST_AUTO_TEST_SUITE_END();
