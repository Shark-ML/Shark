#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <shark/Core/Images/Resize.h>

#define BOOST_TEST_MODULE Core_ImageResize
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <fstream>


#include <shark/Core/Images/ReadImage.h>
#include <shark/Core/Images/WriteImage.h>
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

//downsampling  by a factor 2 in all directions is the average of the 4 pixels
BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_Half){
	RealMatrix result = image::resize(data, shape, {16,13,2}, Interpolation::Linear);
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), 16 * 13 * 2);
	
	for(std::size_t im = 0; im != 3; ++im){
		for(std::size_t i = 0; i != 16; ++i){
			for(std::size_t j = 0; j != 13; ++j){
				double val = 0.25* (data(im, 2 * (2 * i * 26 +2 * j)) + data(im, 2 * ((2 * i + 1) * 26 +2 * j)));
				val +=0.25* (data(im, 2 * (2 * i * 26 +2 * j + 1)) + data(im, 2 * ((2 * i + 1) * 26 +2 * j + 1)));
				BOOST_CHECK_SMALL(result(im, 2 * (i * 13 +j)) - val, 1.e-4);
				
				double val1 = 0.25* (data(im, 2 * (2 * i * 26 +2 * j) + 1) + data(im, 2 * ((2 * i + 1) * 26 +2 * j) + 1));
				val1 +=0.25* (data(im, 2 * (2 * i * 26 +2 * j + 1) + 1) + data(im, 2 * ((2 * i + 1) * 26 +2 * j + 1) + 1));
				BOOST_CHECK_SMALL(result(im, 2 * (i * 13 +j) + 1) - val1 , 1.e-4);
				
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_DoubleX){
	RealMatrix result = image::resize(data, shape, {32,52,2}, Interpolation::Linear);
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), 32 * 52 * 2);
	
	for(std::size_t im = 0; im != 1; ++im){
		for(std::size_t i = 0; i != 32; ++i){
			for(std::size_t j = 0; j != 52; ++j){
				//positions of neighbouring pixels. we guard here for out-ofbounds access
				//for simplicity
				std::size_t j0 = std::size_t((int(j) - 1)/2);
				std::size_t j1 = (j == 0)? 0 : std::min<std::size_t>(j0 + 1, 25);
				//delta values
				double f0 = 0.25 + 0.5  * (j%2);
				double f1 = 1 - f0;
				//apply zero-padding
				if(j == 0)
					f0 = 0;
				if(j == 51)
					f1 = 0;
				
				double val =  f0 * data(im, (i*26+j0) * 2) + f1 * data(im, (i*26+j1) * 2);
				double val1 = f0 * data(im, (i*26+j0) * 2 + 1) + f1 * data(im, (i*26+j1) * 2 + 1);
				BOOST_CHECK_SMALL(result(im, 2 * (i * 52 +j)) - val , 1.e-4);
				BOOST_CHECK_SMALL(result(im, 2 * (i * 52 +j) + 1) - val1 , 1.e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_DoubleY){
	RealMatrix result = image::resize(data, shape, {64,26,2}, Interpolation::Linear);
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), 64 * 26 * 2);
	
	for(std::size_t im = 0; im != 3; ++im){
		for(std::size_t i = 0; i != 64; ++i){
			for(std::size_t j = 0; j != 26; ++j){
				std::size_t i0 = std::size_t((int(i) - 1)/2);
				std::size_t i1 = (i== 0)? 0: std::min<std::size_t>(i0 + 1, 31);
				//delta values
				double f0 = 0.25 + 0.5  * (i%2);
				double f1 = 1 - f0;
				//apply zero-padding
				if(i == 0)
					f0 = 0;
				if(i == 63)
					f1 = 0;
				double val = f0 * data(im, (i0*26+j) * 2) + f1 * data(im, (i1*26+j) * 2);
				double val1 = f0 * data(im, (i0*26+j) * 2 + 1) + f1 * data(im, (i1*26+j) * 2 + 1);
				BOOST_CHECK_SMALL(result(im, 2 * (i * 26 +j)) - val , 1.e-4);
				BOOST_CHECK_SMALL(result(im, 2 * (i * 26 +j) + 1) - val1 , 1.e-4);
			}
		}
	}
}

////////////////////////////DERIVATIVES////////////////////////////////////

BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_Identity_Derivatives){
	RealMatrix result(data.size1(), data.size2());
	image::bilinearResizeWeightedDerivative<double>(data, result, shape, shape);
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), data.size2());
	BOOST_CHECK_SMALL(max(abs(result - data)), 1.e-4);
}

BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_Half_Derivatives){
	RealMatrix half = image::resize(data, shape, {16,13,2});
	RealMatrix result(data.size1(), data.size2());
	image::bilinearResizeWeightedDerivative<double>(half, result, shape, {16,13,2});
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), data.size2());
	
	for(std::size_t im = 0; im != 3; ++im){
		for(std::size_t i = 0; i != 16; ++i){
			for(std::size_t j = 0; j != 13; ++j){
				double val = 0.25 * half(im, 2 * (i * 13 +j));
				double val1 = 0.25 * half(im, 2 * (i * 13 +j) + 1);
				BOOST_CHECK_SMALL( result(im, 2 * (2 * i * 26 +2 * j)) - val , 1.e-4);
				BOOST_CHECK_SMALL( result(im, 2 * (2 * i * 26 +2 * j) + 1) - val1 , 1.e-4);
				BOOST_CHECK_SMALL( result(im, 2 * (2 * i * 26 +2 * j + 1)) - val , 1.e-4);
				BOOST_CHECK_SMALL( result(im, 2 * (2 * i * 26 +2 * j + 1) + 1) - val1 , 1.e-4);
				BOOST_CHECK_SMALL( result(im, 2 * ((2 * i + 1) * 26 +2 * j)) - val , 1.e-4);
				BOOST_CHECK_SMALL( result(im, 2 * ((2 * i + 1) * 26 +2 * j) + 1) - val1 , 1.e-4);
				BOOST_CHECK_SMALL( result(im, 2 * ((2 * i + 1) * 26 +2 * j + 1)) - val , 1.e-4);
				BOOST_CHECK_SMALL( result(im, 2 * ((2 * i + 1) * 26 +2 * j + 1) + 1) - val1 , 1.e-4);
			}
		}
	}
}


BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_DoubleX_Derivatives){
	RealMatrix result(data.size1(), 32 * 13 * 2);
	image::bilinearResizeWeightedDerivative<double>(data, result, {32,13,2}, shape);
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), 32 * 13 * 2);
	
	for(std::size_t im = 0; im != 1; ++im){
		for(std::size_t i = 0; i != 32; ++i){
			for(std::size_t j = 0; j != 13; ++j){
				for(std::size_t c = 0; c != 2; ++c){
					double val = 0.75 * data(im, (i*26+2 * j) * 2 + c);//center-left pixel
					val += 0.75 * data(im, (i*26+2 * j + 1) * 2 + c);//center-right pixel
					if(j > 0)
						val += 0.25 * data(im, (i*26+2 * j - 1) * 2 + c);//left pixel
					if(j < 12)
						val += 0.25 * data(im, (i*26+2 * j + 2) * 2 + c);//right pixel
					BOOST_CHECK_SMALL(result(im, 2 * (i * 13 +j) + c) - val , 1.e-4);
				}
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( Core_Resize_2D_Linear_DoubleY_Derivatives){
	RealMatrix result(data.size1(), 16 * 26 * 2);
	image::bilinearResizeWeightedDerivative<double>(data, result, {16,26,2}, shape);
	BOOST_REQUIRE_EQUAL(result.size1(), data.size1());
	BOOST_REQUIRE_EQUAL(result.size2(), 16 * 26 * 2);
	
	for(std::size_t im = 0; im != 1; ++im){
		for(std::size_t i = 0; i != 16; ++i){
			for(std::size_t j = 0; j != 26; ++j){
				for(std::size_t c = 0; c != 2; ++c){
					double val = 0.75 * data(im, ((2* i)*26+j) * 2 + c);//center-bottom pixel
					val += 0.75 * data(im, ((2* i + 1)*26+j) * 2 + c);//center-top pixel
					if(i > 0)
						val += 0.25 * data(im, ((2* i - 1)*26+j) * 2 + c);//top pixel
					if(i < 15)
						val += 0.25 * data(im, ((2* i + 2)*26+j) * 2 + c);//bottom pixel
					BOOST_CHECK_SMALL(result(im, 2 * (i * 26 +j) + c) - val , 1.e-4);
				}
			}
		}
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
