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
	std::size_t numParams = 2*5*3*6+2;
	Shape imageShape = {11,9,6};
	Shape filterShape = {2,3,5};
	Shape outputShape = {11,9,2};
	Conv2DModel<RealVector, FastSigmoidNeuron> model(imageShape, filterShape);
	BOOST_REQUIRE_EQUAL(model.numberOfParameters(), numParams);
	BOOST_REQUIRE_EQUAL(model.inputShape(), imageShape);
	BOOST_REQUIRE_EQUAL(model.outputShape(), outputShape);
	
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
	
	testWeightedDerivative(model,10,1.e-5,1.e-4);
	testWeightedInputDerivative(model,10,1.e-5, 1.e-5);
	testWeightedDerivativesSame(model,10);
}

BOOST_AUTO_TEST_CASE( Models_Conv2D_Valid)
{
	std::size_t numParams = 2*5*3*6+2;
	Shape imageShape = {11,9,6};
	Shape filterShape = {2,3,5};
	Shape outputShape = {9,5,2};
	Conv2DModel<RealVector, FastSigmoidNeuron> model(imageShape, filterShape, Padding::Valid);
	BOOST_REQUIRE_EQUAL(model.numberOfParameters(), numParams);
	BOOST_REQUIRE_EQUAL(model.inputShape(), imageShape);
	BOOST_REQUIRE_EQUAL(model.outputShape(), outputShape);
	
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
	
	testWeightedDerivative(model,10,1.e-5,1.e-4);
	testWeightedInputDerivative(model,10,1.e-5, 1.e-5);
	testWeightedDerivativesSame(model,10);
}


BOOST_AUTO_TEST_SUITE_END()
