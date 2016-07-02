#define BOOST_TEST_MODULE ML_Download
#include <boost/test/unit_test.hpp>

#include <shark/Data/Download.h>

using namespace shark;


BOOST_AUTO_TEST_SUITE(Data_Download)

BOOST_AUTO_TEST_CASE(Data_Download_URL)
{
	LabeledData<RealVector, unsigned int> dataset;
	downloadSparseData(dataset, "www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide1");
	BOOST_CHECK_EQUAL(dataset.numberOfElements(), 3089);
	BOOST_CHECK_EQUAL(inputDimension(dataset), 4);
	BOOST_CHECK_EQUAL(numberOfClasses(dataset), 2);
}

BOOST_AUTO_TEST_CASE(Data_Download_MLData)
{
	LabeledData<RealVector, unsigned int> dataset;
	downloadFromMLData(dataset, "iris");
	BOOST_CHECK_EQUAL(dataset.numberOfElements(), 150);
	BOOST_CHECK_EQUAL(inputDimension(dataset), 4);
	BOOST_CHECK_EQUAL(numberOfClasses(dataset), 3);
}

BOOST_AUTO_TEST_SUITE_END()
