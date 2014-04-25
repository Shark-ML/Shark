#define BOOST_TEST_MODULE ML_CONCATENATED_MODEL
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <shark/Models/ConcatenatedModel.h>
#include <shark/Models/FFNet.h>
#include <shark/Models/LinearModel.h>
#include <shark/Models/Softmax.h>
#include <shark/Rng/Uniform.h>

#include <sstream>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/polymorphic_text_oarchive.hpp>

using namespace std;
using namespace boost::archive;
using namespace shark;



BOOST_AUTO_TEST_CASE( CONCATENATED_MODEL_Value )
{
	FFNet<LogisticNeuron,LogisticNeuron> net1;
	Softmax net2(2);
	net1.setStructure(3,5,2);
	size_t modelParameters = net1.numberOfParameters();
	ConcatenatedModel<RealVector,RealVector> model (&net1,&net2);

	BOOST_CHECK_EQUAL(model.numberOfParameters(),modelParameters);

	//parameters
	Uniform< Rng::rng_type > uni( shark::Rng::globalRng,-1,1);
	RealVector modelParams(modelParameters);
	RealVector net1Params(net1.numberOfParameters());
	for(size_t i=0;i!=net1Params.size();++i){
		net1Params(i)=uni();
		modelParams(i)=net1Params(i);
	}
	RealVector net2Params(net2.numberOfParameters());
	for(size_t i=0;i!=net2Params.size();++i){
		net2Params(i)=uni();
		modelParams(i+net1Params.size())=net2Params(i);
	}
	//check whether parameter copying is working
	model.setParameterVector(modelParams);
	double error1=norm_sqr(net1Params-net1.parameterVector());
	double error2=norm_sqr(net2Params-net2.parameterVector());
	double error3=norm_sqr(modelParams-model.parameterVector());
	BOOST_CHECK_EQUAL(error1,0.0);
	BOOST_CHECK_EQUAL(error2,0.0);
	BOOST_CHECK_EQUAL(error3,0.0);

	//test Results;
	RealVector input(3);
	for(size_t i=0;i!=3;++i){
		input(i)=uni();
	}
	RealVector intermediateResult = net1(input);
	RealVector endResult = net2(intermediateResult);

	//evaluate point
	RealVector modelResult = model(input);
	double modelError = norm_sqr(modelResult-endResult);
	BOOST_CHECK_SMALL(modelError,1.e-35);
}
BOOST_AUTO_TEST_CASE( CONCATENATED_MODEL_weightedParameterDerivative )
{
	FFNet<LogisticNeuron,LogisticNeuron> net1;
	LinearModel<> net2;
	net1.setStructure(3,5,2);
	net2.setStructure(2,4);
	ConcatenatedModel<RealVector,RealVector> model (&net1,&net2);
	BOOST_CHECK_EQUAL(model.optimizeFirstModelParameters(),1);
	BOOST_CHECK_EQUAL(model.optimizeSecondModelParameters(),1);

	//test1: all activated
	{
		//parameters
		size_t modelParameters = net1.numberOfParameters()+net2.numberOfParameters();
		BOOST_REQUIRE_EQUAL(model.numberOfParameters(), modelParameters);
		RealVector parameters(modelParameters);
		RealVector coefficients(4);
		RealVector point(3);
		for(unsigned int test = 0; test != 10; ++test){
			for(size_t i = 0; i != modelParameters;++i){
				parameters(i) = Rng::uni(-5,5);
			}
			for(size_t i = 0; i != 4;++i){
				coefficients(i) = Rng::uni(-5,5);
			}
			for(size_t i = 0; i != 3;++i){
				point(i) = Rng::uni(-5,5);
			}
			
			model.setParameterVector(parameters);
			testWeightedDerivative(model, point, coefficients, 1.e-5,1.e-8);
		}
	}
	
	//test1: only first model
	{
		//parameters
		size_t modelParameters = net1.numberOfParameters();
		model.optimizeFirstModelParameters() = true;
		model.optimizeSecondModelParameters() = false;
		BOOST_REQUIRE_EQUAL(model.numberOfParameters(), modelParameters);
		RealVector parameters(modelParameters);
		RealVector coefficients(4);
		RealVector point(3);
		for(unsigned int test = 0; test != 10; ++test){
			for(size_t i = 0; i != modelParameters;++i){
				parameters(i) = Rng::uni(-5,5);
			}
			for(size_t i = 0; i != 4;++i){
				coefficients(i) = Rng::uni(-5,5);
			}
			for(size_t i = 0; i != 3;++i){
				point(i) = Rng::uni(-5,5);
			}
			
			model.setParameterVector(parameters);
			testWeightedDerivative(model, point, coefficients, 1.e-5,1.e-8);
		}
	}
	//test2: only second model
	{
		//parameters
		size_t modelParameters = net2.numberOfParameters();
		model.optimizeFirstModelParameters() = false;
		model.optimizeSecondModelParameters() = true;
		BOOST_REQUIRE_EQUAL(model.numberOfParameters(),modelParameters);
		RealVector parameters(modelParameters);
		RealVector coefficients(4);
		RealVector point(3);
		for(unsigned int test = 0; test != 10; ++test){
			for(size_t i = 0; i != modelParameters;++i){
				parameters(i) = Rng::uni(-5,5);
			}
			for(size_t i = 0; i != 4;++i){
				coefficients(i) = Rng::uni(-5,5);
			}
			for(size_t i = 0; i != 3;++i){
				point(i) = Rng::uni(-5,5);
			}
			
			model.setParameterVector(parameters);
			testWeightedDerivative(model, point, coefficients, 1.e-5,1.e-8);
		}
	}
	
	//test3: no parameters
	{
		//parameters
		model.optimizeFirstModelParameters() = false;
		model.optimizeSecondModelParameters() = false;
		BOOST_REQUIRE_EQUAL(model.numberOfParameters(), 0);
	}
}
BOOST_AUTO_TEST_CASE( CONCATENATED_MODEL_weightedInputDerivative )
{
	Softmax net1(10);
	Softmax net2(10);
	size_t modelParameters = net1.numberOfParameters()+net2.numberOfParameters();
	ConcatenatedModel<RealVector,RealVector> model (&net1,&net2);

	RealVector parameters(modelParameters);
	RealVector coefficients(10);
	RealVector point(10);
	for(unsigned int test = 0; test != 100; ++test){
		for(size_t i = 0; i != modelParameters;++i){
			parameters(i) = Rng::uni(-10,10);
		}
		for(size_t i = 0; i != 10;++i){
			coefficients(i) = Rng::uni(-10,10);
		}
		for(size_t i = 0; i != 10;++i){
			point(i) = Rng::uni(-10,10);
		}
		
		model.setParameterVector(parameters);
		testWeightedInputDerivative(model, point, coefficients, 1.e-5,1.e-5);
	}

}
BOOST_AUTO_TEST_CASE( CONCATENATED_MODEL_SERIALIZE )
{
	Softmax net1(10);
	Softmax net2(10);
	ConcatenatedModel<RealVector,RealVector> model (&net1,&net2);

	//parameters
	Uniform<> uni(Rng::globalRng,-1,1);
	RealVector testParameters(model.numberOfParameters());
	for(size_t i=0;i!=model.numberOfParameters();++i){
		testParameters(i)=uni();
	}
	model.setParameterVector(testParameters);

	//the test is, that after deserialization, the results must be identical
	//so we generate some data first
	std::vector<RealVector> data;
	std::vector<RealVector> target;
	RealVector input(10);
	RealVector output(10);

	for (size_t i=0; i<1000; i++)
	{
		for(size_t j=0;j!=10;++j)
		{
			input(j)=Rng::uni(-1,1);
		}
		data.push_back(input);
		target.push_back(model(input));
	}
	RegressionDataset dataset = createLabeledDataFromRange(data,target);

	//now we serialize the model
	ostringstream outputStream;  
	polymorphic_text_oarchive oa(outputStream);  
	oa << model;

	//and create a new model from the serialization
	Softmax netTest1;
	Softmax netTest2;
	ConcatenatedModel<RealVector,RealVector> modelDeserialized (&netTest1,&netTest2);
	istringstream inputStream(outputStream.str());  
	polymorphic_text_iarchive ia(inputStream);
	ia >> modelDeserialized;
	
	//test whether serialization works
	//first simple parameter and topology check
	BOOST_CHECK_SMALL(norm_2(modelDeserialized.parameterVector() - testParameters),1.e-50);
	BOOST_REQUIRE_EQUAL(net1.inputSize(),netTest1.inputSize());
	BOOST_REQUIRE_EQUAL(net2.inputSize(),netTest2.inputSize());
	for (size_t i=0; i<1000; i++)
	{
		RealVector output = modelDeserialized(dataset.element(i).input);
		BOOST_CHECK_SMALL(norm_2(output -dataset.element(i).label),1.e-50);
	}
}
BOOST_AUTO_TEST_CASE( CONCATENATED_MODEL_OPERATOR )
{
	FFNet<LogisticNeuron,LogisticNeuron> net1;
	Softmax net2(2);
	FFNet<LogisticNeuron,LogisticNeuron> net3;
	net1.setStructure(2,5,2);
	net3.setStructure(2,5,2);
	
	size_t modelParameters2 = net1.numberOfParameters()+net2.numberOfParameters();
	size_t modelParameters3 = net1.numberOfParameters()+net2.numberOfParameters()+net3.numberOfParameters();
	
	ConcatenatedModel<RealVector,RealVector> model2 = net1>>net2;
	ConcatenatedModel<RealVector,RealVector> model3 = net1>>net2>>net3;

	BOOST_CHECK_EQUAL(model2.numberOfParameters(),modelParameters2);
	BOOST_CHECK_EQUAL(model3.numberOfParameters(),modelParameters3);

	//parameters
	Uniform< Rng::rng_type > uni( shark::Rng::globalRng,-1,1);
	RealVector modelParams2(modelParameters2);
	RealVector modelParams3(modelParameters3);
	RealVector net1Params(net1.numberOfParameters());
	for(size_t i=0;i!=net1Params.size();++i){
		net1Params(i)=uni();
		modelParams2(i)=net1Params(i);
		modelParams3(i)=net1Params(i);
	}
	RealVector net2Params(net2.numberOfParameters());
	for(size_t i=0;i!=net2Params.size();++i){
		net2Params(i)=uni();
		modelParams2(i+net1Params.size())=net2Params(i);
		modelParams3(i+net1Params.size())=net2Params(i);
	}
	RealVector net3Params(net3.numberOfParameters());
	for(size_t i=0;i!=net3Params.size();++i){
		net3Params(i)=uni();
		modelParams3(i+modelParameters2)=net3Params(i);
	}
	//check whether parameter copying is working
	
	//two models
	model2.setParameterVector(modelParams2);
	double error1=norm_sqr(net1Params-net1.parameterVector());
	double error2=norm_sqr(net2Params-net2.parameterVector());
	double errorComplete=norm_sqr(modelParams2-model2.parameterVector());
	
	BOOST_CHECK_EQUAL(error1,0.0);
	BOOST_CHECK_EQUAL(error2,0.0);
	BOOST_CHECK_EQUAL(errorComplete,0.0);
	
	
	//clear parameters
	initRandomNormal(net1,1);
	initRandomNormal(net2,1);
	initRandomNormal(net3,1);
	
	//three models
	model3.setParameterVector(modelParams3);
	error1=norm_sqr(net1Params-net1.parameterVector());
	error2=norm_sqr(net2Params-net2.parameterVector());
	double error3=norm_sqr(net3Params-net3.parameterVector());
	errorComplete=norm_sqr(modelParams3-model3.parameterVector());
	
	BOOST_CHECK_EQUAL(error1,0.0);
	BOOST_CHECK_EQUAL(error2,0.0);
	BOOST_CHECK_EQUAL(error3,0.0);
	BOOST_CHECK_EQUAL(errorComplete,0.0);

	//test Results;
	for(std::size_t trial = 0; trial != 1000; ++trial){
		RealVector input(2);
		for(size_t i=0;i!=2;++i){
			input(i)=uni();
		}
		RealVector intermediateResult1 = net1(input);
		RealVector intermediateResult2 = net2(intermediateResult1);
		RealVector endResult = net3(intermediateResult2);
		//evaluate point
		RealVector modelResult2 = model2(input);
		RealVector modelResult3 = model3(input);
		double modelError2 = norm_sqr(modelResult2-intermediateResult2);
		double modelError3 = norm_sqr(modelResult3-endResult);
		BOOST_CHECK_SMALL(modelError2,1.e-35);
		BOOST_CHECK_SMALL(modelError3,1.e-35);
	}
}
