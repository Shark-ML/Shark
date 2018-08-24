#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <shark/Core/Images/Resize.h>

#define BOOST_TEST_MODULE Core_ImageResize
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <fstream>
using namespace shark;
using namespace std;

struct ImageFixture {
	ImageFixture()
	: shape({32,26,2}), data(3,32*26 * 2){
		for(std::size_t im = 0; im != 3; ++im){
			for(std::size_t i = 0; i != 32; ++i){
				for(std::size_t j = 0; j != 26; ++j){
					double G = 0.0;
					if( i >= 16)
						G += 1.0/3.0;
					if(j >= 13)
						G += 1.0/3.0;
					data(im, (i*26+j) * 2) = G;
					data(im, (i*26+j) * 2 + 1) = i % 2 == 0;
				}
			}
		}
	}

	Shape shape;
	RealMatrix data;
};


BOOST_FIXTURE_TEST_SUITE (Core_ImageResize_Tests, ImageFixture )

BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_Identity){
	RealMatrix result = image::resize(data, shape, shape, Interpolation::Linear);
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), data.size2());
	
	BOOST_CHECK_SMALL(max(abs(result - data)), 1.e-4);
}

BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_Half){
	RealMatrix result = image::resize(data, shape, {16,13,2}, Interpolation::Linear);
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), 16 * 13 * 2);
	
	for(std::size_t im = 0; im != 3; ++im){
		for(std::size_t i = 0; i != 16; ++i){
			for(std::size_t j = 0; j != 13; ++j){
				BOOST_CHECK_SMALL(result(im, 2 * (i * 13 +j)) - data(im, 2 * (2 * i * 26 +2 * j)) , 1.e-4);
				BOOST_CHECK_SMALL(result(im, 2 * (i * 13 +j) + 1) - data(im, 2 * (2 * i * 26 +2 * j) + 1) , 1.e-4);
				
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_DoubleX){
	RealMatrix result = image::resize(data, shape, {32,52,2}, Interpolation::Linear);
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), 32 * 52 * 2);
	
	for(std::size_t im = 0; im != 3; ++im){
		for(std::size_t i = 0; i != 32; ++i){
			for(std::size_t j = 0; j != 26; ++j){
				double val = data(im, (i*26+j) * 2);
				double val1 = data(im, (i*26+j) * 2 + 1);
				BOOST_CHECK_SMALL(result(im, 2 * (i * 52 +2 * j)) - val , 1.e-4);
				BOOST_CHECK_SMALL(result(im, 2 * (i * 52 +2 * j) + 1) - val1 , 1.e-4);
				
				val *= 0.5;
				val1 *= 0.5;
				if(j < 25){
					val += 0.5 * data(im, (i*26+j + 1) * 2);
					val1+= 0.5 * data(im, (i*26+j + 1) * 2 + 1);
				}
				BOOST_CHECK_SMALL(result(im, 2 * (i * 52 +2 * j + 1)) - val , 1.e-4);
				BOOST_CHECK_SMALL(result(im, 2 * (i * 52 +2 * j + 1) + 1) - val1, 1.e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_DoubleY){
	RealMatrix result = image::resize(data, shape, {64,26,2}, Interpolation::Linear);
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), 32 * 52 * 2);
	
	for(std::size_t im = 0; im != 3; ++im){
		for(std::size_t i = 0; i != 32; ++i){
			for(std::size_t j = 0; j != 26; ++j){
				double val = data(im, (i*26+j) * 2);
				double val1 = data(im, (i*26+j) * 2 + 1);
				BOOST_CHECK_SMALL(result(im, 2 * (i * 52 +j)) - val , 1.e-4);
				BOOST_CHECK_SMALL(result(im, 2 * (i * 52 +j) + 1) - val1 , 1.e-4);
				
				val *= 0.5;
				val1 *= 0.5;
				if(i < 31){
					val += 0.5 * data(im, ((i+1)*26+j) * 2);
					val1+= 0.5 * data(im, ((i+1)*26+j) * 2 + 1);
				}
				BOOST_CHECK_SMALL(result(im, 2 * ((2* i +1) * 26 +j)) - val , 1.e-4);
				BOOST_CHECK_SMALL(result(im, 2 * ((2* i +1) * 26 +j) + 1) - val1, 1.e-4);
			}
		}
	}
}

////////////////////////////DERIVATIVES////////////////////////////////////

BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_Identity_Derivatives){
	RealMatrix result(data.size1(), data.size2());
	image::resizeWeightedDerivative(data, result, shape, shape, Interpolation::Linear);
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), data.size2());
	BOOST_CHECK_SMALL(max(abs(result - data)), 1.e-4);
}

BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_Half_Derivatives){
	RealMatrix half = image::resize(data, shape, {16,13,2}, Interpolation::Linear);
	RealMatrix result(data.size1(), data.size2());
	image::resizeWeightedDerivative(half, result, shape, {16,13,2});
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), data.size2());
	
	//every second row and every second column must be 0, otherwise the value must be exactly the backpropagated valeus
	for(std::size_t im = 0; im != 3; ++im){
		for(std::size_t i = 0; i != 16; ++i){
			for(std::size_t j = 0; j != 13; ++j){
				BOOST_CHECK_SMALL(half(im, 2 * (i * 13 +j)) - result(im, 2 * (2 * i * 26 +2 * j)) , 1.e-4);
				BOOST_CHECK_SMALL(half(im, 2 * (i * 13 +j) + 1) - result(im, 2 * (2 * i * 26 +2 * j) + 1) , 1.e-4);
				BOOST_CHECK_SMALL(result(im, 2 * (2 * i * 26 +2 * j + 1)) , 1.e-4);
				BOOST_CHECK_SMALL(result(im, 2 * (2 * i * 26 +2 * j + 1) + 1) , 1.e-4);
				BOOST_CHECK_SMALL(result(im, 2 * ((2 * i + 1) * 26 +2 * j)) , 1.e-4);
				BOOST_CHECK_SMALL(result(im, 2 * ((2 * i + 1) * 26 +2 * j) + 1) , 1.e-4);
				BOOST_CHECK_SMALL(result(im, 2 * ((2 * i + 1) * 26 +2 * j + 1)) , 1.e-4);
				BOOST_CHECK_SMALL(result(im, 2 * ((2 * i + 1) * 26 +2 * j + 1) + 1) , 1.e-4);
			}
		}
	}
}


BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_DoubleX_Derivatives){
	RealMatrix result(data.size1(), 32 * 13 * 2);
	image::resizeWeightedDerivative(data, result, {32,13,2}, shape, Interpolation::Linear);
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), 32 * 13 * 2);
	
	//derivative must be equal to c_ij+0.5 * c_i{j-1} + +0.5 * c_i{j+1}
	for(std::size_t im = 0; im != 3; ++im){
		for(std::size_t i = 0; i != 32; ++i){
			for(std::size_t j = 0; j != 13; ++j){
				double val = data(im, (i*26+2 * j) * 2);//center pixel
				val += 0.5 * data(im, (i*26+2 * j + 1) * 2);//right pixel
				if(j > 0)
					val += 0.5 * data(im, (i*26+2 * j - 1) * 2);//left pixel if possible
				BOOST_CHECK_SMALL(result(im, 2 * (i * 13 +j)) - val , 1.e-4);
				
				double val1 = data(im, (i*26+2 * j) * 2 + 1);//center pixel
				val1 += 0.5 * data(im, (i*26+2 * j + 1) * 2 + 1);//right pixel
				if(j > 0)
					val1 += 0.5 * data(im, (i*26+2 * j - 1) * 2 + 1);//left pixel if possible
				BOOST_CHECK_SMALL(result(im, 2 * (i * 13 +j) + 1) - val1 , 1.e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_DoubleY_Derivatives){
	RealMatrix result(data.size1(), 16 * 26 * 2);
	image::resizeWeightedDerivative(data, result, {16,26,2}, shape, Interpolation::Linear);
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), 16 * 26 * 2);
	
	//derivative must be equal to c_ij+0.5 * c_i{j-1} + +0.5 * c_i{j+1}
	for(std::size_t im = 0; im != 3; ++im){
		for(std::size_t i = 0; i != 16; ++i){
			for(std::size_t j = 0; j != 26; ++j){
				double val = data(im, (i * 2 * 26+j) * 2);//center pixel
				val += 0.5 * data(im, ((i * 2 + 1)*26+j) * 2);//bottom pixel
				if(i > 0)
					val += 0.5 * data(im, ((i * 2 - 1)*26 + j) * 2);//top pixel if possible
				BOOST_CHECK_SMALL(result(im, 2 * (i * 26 +j)) - val , 1.e-4);
				
				double val1 = data(im, (i * 2 * 26+j) * 2 + 1);//center pixel
				val1 += 0.5 * data(im, ((i * 2 + 1)*26+j) * 2 + 1);//bottom pixel
				if(i > 0)
					val1 += 0.5 * data(im, ((i * 2 - 1)*26 + j) * 2 + 1);//top pixel if possible
				BOOST_CHECK_SMALL(result(im, 2 * (i * 26 +j) + 1) - val1 , 1.e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
