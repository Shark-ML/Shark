
#define BOOST_TEST_MODULE Core_ZipSupport
#include <shark/Core/ZipSupport.h>

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <iostream>
#include <algorithm>

BOOST_AUTO_TEST_SUITE (CoreZipSupport_Tests)

BOOST_AUTO_TEST_CASE( Core_Read_Test){
	shark::ZipReader reader( "Test/test_data/test_zip.zip");
	
	std::vector<std::string> filenames = { "zip/foo.txt", "zip/foobar.txt"};
	BOOST_REQUIRE_EQUAL(reader.numFiles(), filenames.size());
	
	std::vector<std::string> testFiles;
	for(std::size_t i = 0; i != reader.numFiles(); ++i){
		testFiles.push_back(reader.fileName(i));
	}
	std::sort(testFiles.begin(), testFiles.end());
	for(std::size_t i = 0; i != reader.numFiles(); ++i){
		BOOST_CHECK_EQUAL(testFiles[i], filenames[i]);
	}
	
	testFiles = reader.fileNames();
	std::sort(testFiles.begin(), testFiles.end());
	for(std::size_t i = 0; i != reader.numFiles(); ++i){
		BOOST_CHECK_EQUAL(testFiles[i], filenames[i]);
	}
	//check correct file contents
	{
		std::vector<unsigned char> result = reader.readFile(filenames[0]);
		std::vector<unsigned char> truth = {(unsigned char)'f', (unsigned char)'o', (unsigned char)'o', (unsigned char)'\n'};
		BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), truth.begin(), truth.end());
	}
	
	{
		std::vector<unsigned char> result = reader.readFile(filenames[1]);
		std::vector<unsigned char> truth = {(unsigned char)'f', (unsigned char)'o', (unsigned char)'o', (unsigned char)'b', (unsigned char)'a', (unsigned char)'r', (unsigned char)'\n'};
		BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), truth.begin(), truth.end());
	}
	//check that lookup by name gives same result as lookup by index
	{
		auto result = reader.readFile(0); 
		auto resultName = reader.readFile(reader.fileName(0));
		BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), resultName.begin(), resultName.end());
	}
	
	{
		auto result = reader.readFile(1); 
		auto resultName = reader.readFile(reader.fileName(1));
		BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), resultName.begin(), resultName.end());
	}
}

BOOST_AUTO_TEST_SUITE_END()
