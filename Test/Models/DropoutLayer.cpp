#define BOOST_TEST_MODULE MODEL_DROPOUTLAYER
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <shark/Models/DropoutLayer.h>

using namespace std;
using namespace boost::archive;
using namespace shark;

//check that the structure is correct, i.e. matrice have the right form and setting parameters works
BOOST_AUTO_TEST_SUITE (Models_DropoutLayer)


BOOST_AUTO_TEST_CASE( DropoutLayer_Value)
{
	DropoutLayer<> layer(10,0.5,random::globalRng);
	RealMatrix inputs(100,10);
	for(std::size_t i = 0; i != 100; ++i){
		for(std::size_t j = 0; j != 10; ++j){
			inputs(i,j) = random::uni(random::globalRng,2,3);
		}
	}
	
	auto state = layer.createState();
	RealMatrix outputs = layer(inputs);
	RealMatrix outputs2;
	layer.eval(inputs,outputs2, *state);
	for(std::size_t j = 0; j != 10; ++j){
		int count = 0;
		int count2 = 0;
		for(std::size_t i = 0; i != 100; ++i){
			BOOST_CHECK(outputs(i,j) == 0.0 || outputs(i,j) == inputs(i,j));
			BOOST_CHECK(outputs2(i,j) == 0.0 || outputs2(i,j) == inputs(i,j));
			if(outputs(i,j) == 0.0) ++count;
			if(outputs2(i,j) == 0.0) ++count2;
		}
		
		BOOST_CHECK(count > 30 && count < 70);
		BOOST_CHECK(count2 > 30 && count2 < 70);
	}
	
}

BOOST_AUTO_TEST_CASE( DropoutLayer_Derivative)
{
	DropoutLayer<> layer(10,0.5,random::globalRng);
	RealMatrix inputs(100,10);
	RealMatrix coeffs(100,10);
	for(std::size_t i = 0; i != 100; ++i){
		for(std::size_t j = 0; j != 10; ++j){
			inputs(i,j) = random::uni(random::globalRng,2,3);
			coeffs(i,j) = random::gauss(random::globalRng,2,3);
		}
	}
	
	
	for(std::size_t i = 0; i != 100; ++i){
		for(std::size_t j = 0; j != 10; ++j){
			inputs(i,j) = random::uni(random::globalRng,2,3);
		}
	}
	
	auto state = layer.createState();
	RealMatrix outputs;
	layer.eval(inputs,outputs, *state);
	RealMatrix derivative;
	layer.weightedInputDerivative(inputs,outputs,coeffs, *state, derivative);
	
	for(std::size_t i = 0; i != 100; ++i){
		for(std::size_t j = 0; j != 10; ++j){
			BOOST_CHECK(derivative(i,j) == 0.0 || derivative(i,j) == coeffs(i,j));
		}
	}
	
}
BOOST_AUTO_TEST_SUITE_END()
