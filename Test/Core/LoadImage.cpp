#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <shark/Core/Images/LoadImage.h>

#define BOOST_TEST_MODULE Core_ImageReorder
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <fstream>
using namespace shark;
using namespace std;

struct ImageFixture {
	ImageFixture(): shape({32,26,4}), data(32*26*4,0){
		//red and green channels are 0
		for(std::size_t i = 0; i != 32; ++i){
			for(std::size_t j = 0; j != 26; ++j){
				data[(i*26+j)*4+2] = 1.0;//blue channel is maximal
				double alpha = 0.0;
				if( i >= 16)
					alpha += 2.0/3.0;
				if(j >= 13)
					alpha += 1.0/3.0;
				data[(i*26+j)*4+3] = alpha;//alpha channel are 4 squares
			}
		}
	}

	Shape shape;
	RealVector data;
};


BOOST_FIXTURE_TEST_SUITE (Core_LoadImage_Tests, ImageFixture )

BOOST_AUTO_TEST_CASE( Core_Load_PNG){
	std::ifstream file( "./test_data/testImage.png", std::ios::binary );
	if(!file)
		std::cout<<"could not open file"<<std::endl;
	file.seekg(0, std::ios::end);
	std::streampos fileSize = file.tellg();
	file.seekg(0, std::ios::beg);
	std::vector<unsigned char> buffer(fileSize);
	file.read((char*) &buffer[0], fileSize);

	
	std::pair<blas::vector<double>, Shape> result = image::readPNG<double>(buffer);
	
	BOOST_REQUIRE_EQUAL(result.second[0], shape[0]);
	BOOST_REQUIRE_EQUAL(result.second[1], shape[1]);
	BOOST_REQUIRE_EQUAL(result.second[2], shape[2]);
	BOOST_REQUIRE_EQUAL(result.first.size(), data.size());
	
	for(std::size_t i = 0; i != data.size(); ++i){
		BOOST_CHECK_SMALL(result.first[i] - data[i], 1.e-4);
	}
}


BOOST_AUTO_TEST_SUITE_END()
