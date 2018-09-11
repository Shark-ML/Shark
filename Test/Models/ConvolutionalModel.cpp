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
	Shape imageShape = {6,11,9};
	Shape filterShape = {2,3,5};
	Shape outputShape = {2,11,9};
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

BOOST_AUTO_TEST_CASE( Models_Conv2D_Linear)
{
	std::size_t numParams = 2*5*3*6+2;
	Shape imageShape = {6,11,9};
	Shape filterShape = {2,3,5};
	Shape outputShape = {2,11,9};
	Conv2DModel<RealVector, LinearNeuron> model(imageShape, filterShape);
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
	Shape imageShape = {6,11,9};
	Shape filterShape = {2,3,5};
	Shape outputShape = {2, 9,5};
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

BOOST_AUTO_TEST_CASE( Models_Conv2D_Valid_Linear)
{
	std::size_t numParams = 2*5*3*6+2;
	Shape imageShape = {6,11,9};
	Shape filterShape = {2,3,5};
	Shape outputShape = {2, 9,5};
	Conv2DModel<RealVector, LinearNeuron> model(imageShape, filterShape, Padding::Valid);
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

#if defined(__HCC__) || defined(__NVCC__)
BOOST_AUTO_TEST_CASE( Models_Conv2D_HIP){
	std::size_t numParams = 2*5*3*6+2;
	Shape imageShape = {6,11,9};
	Shape filterShape = {2,3,5};
	Shape outputShape = {2,11,9};
	typedef blas::vector<float, blas::hip_tag> VectorTypeHIP;
	typedef blas::vector<float> VectorType;
	typedef blas::matrix<float, blas::row_major, blas::hip_tag> MatrixTypeHIP;
	typedef blas::matrix<float> MatrixType;
	Conv2DModel<VectorTypeHIP, FastSigmoidNeuron> modelHIP(imageShape, filterShape);
	Conv2DModel<VectorType, FastSigmoidNeuron> model(imageShape, filterShape);
	BOOST_REQUIRE_EQUAL(modelHIP.numberOfParameters(), numParams);
	BOOST_REQUIRE_EQUAL(modelHIP.inputShape(), imageShape);
	BOOST_REQUIRE_EQUAL(modelHIP.outputShape(), outputShape);
	
	//check parameter vector
	{
		VectorType params(numParams);
		std::iota(params.begin(),params.end(),1);
		params /= params.size();
		//check that setting and retrieving works
		model.setParameterVector(params);
		modelHIP.setParameterVector(blas::copy_to_device(params, blas::hip_tag()));
		VectorType paramsTest = blas::copy_to_cpu(modelHIP.parameterVector());
		BOOST_REQUIRE_EQUAL(paramsTest.size(), numParams);
		for(std::size_t i = 0; i != paramsTest.size(); ++i){
			BOOST_CHECK_CLOSE(params(i),paramsTest(i),1.e-10);
		}
	}
	
	for(std::size_t t = 0; t != 100; ++t){
		MatrixType inputs = normal(random::globalRng(), 5, imageShape.numElements(), 0.0f, 1.0f, blas::cpu_tag());
		MatrixType coefficients = normal(random::globalRng(), 5, outputShape.numElements(), 0.0f, 1.0f, blas::cpu_tag());
		auto state = model.createState();
		
		MatrixTypeHIP inputsHIP = blas::copy_to_device(inputs, blas::hip_tag());
		MatrixTypeHIP coefficientsHIP = blas::copy_to_device(coefficients, blas::hip_tag());
		auto stateHIP = modelHIP.createState();
		
		MatrixType evalOutput;
		MatrixTypeHIP evalOutputHIP;
		model.eval(inputs, evalOutput, *state);
		modelHIP.eval(inputsHIP, evalOutputHIP, *stateHIP);
		BOOST_CHECK_SMALL(max(abs(evalOutput - blas::copy_to_cpu(evalOutputHIP))),1.e-5f);
		
		VectorType gradient;
		VectorTypeHIP gradientHIP;
		model.weightedParameterDerivative(inputs, evalOutput, coefficients, *state, gradient);
		modelHIP.weightedParameterDerivative(inputsHIP, evalOutputHIP, coefficientsHIP, *stateHIP, gradientHIP);
		BOOST_CHECK_SMALL(norm_inf(gradient - blas::copy_to_cpu(gradientHIP)),1.e-4f);
		
		MatrixType gradInput;
		MatrixTypeHIP gradInputHIP;
		model.weightedInputDerivative(inputs, evalOutput, coefficients, *state, gradInput);
		modelHIP.weightedInputDerivative(inputsHIP, evalOutputHIP, coefficientsHIP, *stateHIP, gradInputHIP);
		
		BOOST_CHECK_SMALL(max(abs(gradInput - blas::copy_to_cpu(gradInput))),1.e-4f);
	}
}


BOOST_AUTO_TEST_CASE( Models_Conv2D_Valid_HIP)
{
	std::size_t numParams = 2*5*3*6+2;
	Shape imageShape = {6,11,9};
	Shape filterShape = {2,3,5};
	Shape outputShape = {2, 9,5};
	typedef blas::vector<float, blas::hip_tag> VectorTypeHIP;
	typedef blas::vector<float> VectorType;
	typedef blas::matrix<float, blas::row_major, blas::hip_tag> MatrixTypeHIP;
	typedef blas::matrix<float> MatrixType;
	Conv2DModel<VectorTypeHIP, FastSigmoidNeuron> modelHIP(imageShape, filterShape, Padding::Valid);
	Conv2DModel<VectorType, FastSigmoidNeuron> model(imageShape, filterShape, Padding::Valid);
	BOOST_REQUIRE_EQUAL(model.numberOfParameters(), numParams);
	BOOST_REQUIRE_EQUAL(model.inputShape(), imageShape);
	BOOST_REQUIRE_EQUAL(model.outputShape(), outputShape);
	
	//check parameter vector
	{
		VectorType params(numParams);
		std::iota(params.begin(),params.end(),1);
		params /= params.size();
		//check that setting and retrieving works
		model.setParameterVector(params);
		modelHIP.setParameterVector(blas::copy_to_device(params, blas::hip_tag()));
		VectorType paramsTest = blas::copy_to_cpu(modelHIP.parameterVector());
		BOOST_REQUIRE_EQUAL(paramsTest.size(), numParams);
		for(std::size_t i = 0; i != paramsTest.size(); ++i){
			BOOST_CHECK_CLOSE(params(i),paramsTest(i),1.e-10);
		}
	}
	
	for(std::size_t t = 0; t != 100; ++t){
		MatrixType inputs = normal(random::globalRng(), 5, imageShape.numElements(), 0.0f, 1.0f, blas::cpu_tag());
		MatrixType coefficients = normal(random::globalRng(), 5, outputShape.numElements(), 0.0f, 1.0f, blas::cpu_tag());
		auto state = model.createState();
		
		MatrixTypeHIP inputsHIP = blas::copy_to_device(inputs, blas::hip_tag());
		MatrixTypeHIP coefficientsHIP = blas::copy_to_device(coefficients, blas::hip_tag());
		auto stateHIP = modelHIP.createState();
		
		MatrixType evalOutput;
		MatrixTypeHIP evalOutputHIP;
		model.eval(inputs, evalOutput, *state);
		modelHIP.eval(inputsHIP, evalOutputHIP, *stateHIP);
		BOOST_CHECK_SMALL(max(abs(evalOutput - blas::copy_to_cpu(evalOutputHIP))),1.e-5f);
		
		VectorType gradient;
		VectorTypeHIP gradientHIP;
		model.weightedParameterDerivative(inputs, evalOutput, coefficients, *state, gradient);
		modelHIP.weightedParameterDerivative(inputsHIP, evalOutputHIP, coefficientsHIP, *stateHIP, gradientHIP);
		BOOST_CHECK_SMALL(norm_inf(gradient - blas::copy_to_cpu(gradientHIP)),1.e-4f);
		
		MatrixType gradInput;
		MatrixTypeHIP gradInputHIP;
		model.weightedInputDerivative(inputs, evalOutput, coefficients, *state, gradInput);
		modelHIP.weightedInputDerivative(inputsHIP, evalOutputHIP, coefficientsHIP, *stateHIP, gradInputHIP);
		
		BOOST_CHECK_SMALL(max(abs(gradInput - blas::copy_to_cpu(gradInput))),1.e-4f);
	}
}
#endif


BOOST_AUTO_TEST_SUITE_END()
