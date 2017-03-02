#define BOOST_TEST_MODULE ML_Download
#include <boost/test/unit_test.hpp>

#include <shark/Data/Download.h>
#include <iostream>

using namespace shark;


bool verifyConnection()
{
	detail::Socket socket("mldata.org", 80);
	return socket.connected();
}


BOOST_AUTO_TEST_SUITE(Data_Download)

BOOST_AUTO_TEST_CASE(Data_Download_URL)
{
	if (! verifyConnection())
	{
		std::cout << "cannot reach mldata.org server; skipping data download test" << std::endl;
		return;
	}

	// test the download of a data file from a given URL
	LabeledData<RealVector, unsigned int> dataset;
	downloadSparseData(dataset, "http://mldata.org/repository/data/download/libsvm/iris/");
	BOOST_CHECK_EQUAL(dataset.numberOfElements(), 150);
	BOOST_CHECK_EQUAL(inputDimension(dataset), 4);
	BOOST_CHECK_EQUAL(numberOfClasses(dataset), 3);
}

BOOST_AUTO_TEST_CASE(Data_Download_MLData)
{
	if (! verifyConnection())
	{
		std::cout << "cannot reach mldata.org server; skipping data download test" << std::endl;
		return;
	}

	// test the download of a data file from openml.org given a data set name
	LabeledData<RealVector, unsigned int> dataset;
	downloadFromMLData(dataset, "iris");
	BOOST_CHECK_EQUAL(dataset.numberOfElements(), 150);
	BOOST_CHECK_EQUAL(inputDimension(dataset), 4);
	BOOST_CHECK_EQUAL(numberOfClasses(dataset), 3);
}

BOOST_AUTO_TEST_SUITE_END()
