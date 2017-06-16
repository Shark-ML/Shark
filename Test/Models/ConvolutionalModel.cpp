#include <shark/Models/ConvolutionalModel.h>
#define BOOST_TEST_MODULE Models_ConvolutionalModel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <algorithm>

using namespace shark;


BOOST_AUTO_TEST_SUITE (Models_ConvolutionalModel)

BOOST_AUTO_TEST_CASE( Models_Conv2D)
{
	std::size_t numParams = 6*5*3*4+6;
	Conv2DModel<RealVector, FastSigmoidNeuron> model({10,9}, {3,4}, 5, 6);
	BOOST_REQUIRE_EQUAL(model.numberOfParameters(), numParams);
	BOOST_REQUIRE_EQUAL(model.inputSize(), 5*10*9);
	BOOST_REQUIRE_EQUAL(model.outputSize(), 6*8*6);
	
	//check parameter vector
	{
		RealVector params(numParams);
		std::iota(params.begin(),params.end(),1);
		params /= params.size();
		//check that setting and retrieving works
		model.setParameterVector(params);
		RealVector paramsTest = model.parameterVector();
		BOOST_REQUIRE_EQUAL(paramsTest.size(), numParams);
		for(std::size_t i = 0; i != paramsTest.size(); ++i){
			BOOST_CHECK_CLOSE(params(i),paramsTest(i),1.e-15);
		}
	}
	
	testWeightedDerivative(model,10);
	testWeightedInputDerivative(model,10,1.e-5, 1.e-7);
	testWeightedDerivativesSame(model,10);
}


BOOST_AUTO_TEST_SUITE_END()
