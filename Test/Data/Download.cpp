#define BOOST_TEST_MODULE ML_Download
#include <boost/test/unit_test.hpp>

#include <shark/Data/Download.h>
#include <iostream>

using namespace shark;


bool verifyConnection()
{
	std::string domain;
	std::tie(domain, std::ignore) = splitUrl("http://mldata.org/repository/data/download/libsvm/iris/");
	detail::Socket socket(domain, 80);
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

BOOST_AUTO_TEST_CASE(Data_Download_Url_splitter)
{
	std::vector<std::string> urls{
		"http://mldata.org/repository/data/download/libsvm/iris/",
			"http://dr.dk/nyhederne",
		    "google.com/en?sdfsdfsfs",
			"https://secret.website.com/noaccess"};
	std::vector<std::pair<std::string, std::string>> exp{
		std::make_pair("mldata.org", "/repository/data/download/libsvm/iris/"),
		std::make_pair("dr.dk", "/nyhederne"),
		std::make_pair("google.com", "/en?sdfsdfsfs"),
		std::make_pair("secret.website.com", "/noaccess")
			};
	for(std::size_t i = 0; i < urls.size(); ++i)
	{
		std::string d, r;
		std::tie(d, r) = splitUrl(urls[i]);
		BOOST_CHECK_EQUAL(d, std::get<0>(exp[i]));
		BOOST_CHECK_EQUAL(r, std::get<1>(exp[i]));
	}
}

BOOST_AUTO_TEST_SUITE_END()
