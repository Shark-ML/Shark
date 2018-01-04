#define BOOST_TEST_MODULE ML_CONCATENATED_MODEL
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <shark/Models/ConcatenatedModel.h>
#include <shark/Models/LinearModel.h>
#include <shark/Models/NeuronLayers.h>

#include <sstream>



using namespace std;
using namespace boost::archive;
using namespace shark;



BOOST_AUTO_TEST_SUITE (Models_ConcatenatedModel)

BOOST_AUTO_TEST_CASE( CONCATENATED_MODEL_Value )
{
	LinearModel<> net1;
	NeuronLayer<SoftmaxNeuron<> > net2(5);
	LinearModel<> net3;
	net1.setStructure(3,5,1);
	net3.setStructure(5,2,1);
	size_t modelParameters = net1.numberOfParameters()+net3.numberOfParameters();
	ConcatenatedModel<RealVector> model  = net1 >> net2 >> net3;

	BOOST_CHECK_EQUAL(model.numberOfParameters(),modelParameters);

	//parameters
	
	RealVector modelParams(modelParameters);
	RealVector net1Params(net1.numberOfParameters());
	for(size_t i=0;i!=net1Params.size();++i){
		net1Params(i)=shark::random::uni( shark::random::globalRng,-1,1);
		modelParams(i)=net1Params(i);
	}
	RealVector net2Params(net2.numberOfParameters());
	for(size_t i=0;i!=net2Params.size();++i){
		net2Params(i)=shark::random::uni( shark::random::globalRng,-1,1);
		modelParams(i+net1Params.size())=net2Params(i);
	}
	RealVector net3Params(net3.numberOfParameters());
	for(size_t i=0;i!=net3Params.size();++i){
		net3Params(i)=shark::random::uni( shark::random::globalRng,-1,1);
		modelParams(i+net1Params.size() + net2Params.size())=net3Params(i);
	}
	//check whether parameter copying is working
	model.setParameterVector(modelParams);
	double error1=norm_sqr(net1Params-net1.parameterVector());
	double error2=norm_sqr(net2Params-net2.parameterVector());
	double error3=norm_sqr(net3Params-net3.parameterVector());
	double error4=norm_sqr(modelParams-model.parameterVector());
	BOOST_CHECK_EQUAL(error1,0.0);
	BOOST_CHECK_EQUAL(error2,0.0);
	BOOST_CHECK_EQUAL(error3,0.0);
	BOOST_CHECK_EQUAL(error4,0.0);

	//test Results;
	RealVector input(3);
	for(size_t i=0;i!=3;++i){
		input(i)=shark::random::uni( shark::random::globalRng,-1,1);
	}
	RealVector intermediateResult1 = net1(input);
	RealVector intermediateResult2 = net2(intermediateResult1);
	RealVector endResult = net3(intermediateResult2);

	//evaluate point
	RealVector modelResult = model(input);
	double modelError = norm_sqr(modelResult-endResult);
	BOOST_CHECK_SMALL(modelError,1.e-14);
}
BOOST_AUTO_TEST_CASE( CONCATENATED_MODEL_weightedParameterDerivative )
{
	LinearModel<> net1;
	NeuronLayer<SoftmaxNeuron<> > net2(5);
	LinearModel<> net3;
	net1.setStructure(3,5,1);
	net3.setStructure(5,2,1);
	ConcatenatedModel<RealVector> model  = net1 >> net2 >> net3;

	//test1: all activated
	{
		//parameters(net2 doe snot have parameters)
		size_t modelParameters = net1.numberOfParameters()+net3.numberOfParameters();
		BOOST_REQUIRE_EQUAL(model.numberOfParameters(), modelParameters);
		RealVector parameters(modelParameters);
		RealVector coefficients(2);
		RealVector point(3);
		for(unsigned int test = 0; test != 10; ++test){
			for(size_t i = 0; i != modelParameters;++i){
				parameters(i) = random::uni(random::globalRng,-5,5);
			}
			for(size_t i = 0; i != 2;++i){
				coefficients(i) = random::uni(random::globalRng,-5,5);
			}
			for(size_t i = 0; i != 3;++i){
				point(i) = random::uni(random::globalRng,-5,5);
			}
			
			model.setParameterVector(parameters);
			testWeightedDerivative(model, point, coefficients, 1.e-5,1.e-8);
		}
	}
	
	//test1: only first model
	{
		//parameters
		size_t modelParameters = net1.numberOfParameters();
		model.enableModelOptimization(0,true);
		model.enableModelOptimization(1,false);
		model.enableModelOptimization(2,false);
		BOOST_REQUIRE_EQUAL(model.numberOfParameters(), modelParameters);
		RealVector parameters(modelParameters);
		RealVector coefficients(2);
		RealVector point(3);
		for(unsigned int test = 0; test != 10; ++test){
			for(size_t i = 0; i != modelParameters;++i){
				parameters(i) = random::uni(random::globalRng,-5,5);
			}
			for(size_t i = 0; i != 2;++i){
				coefficients(i) = random::uni(random::globalRng,-5,5);
			}
			for(size_t i = 0; i != 3;++i){
				point(i) = random::uni(random::globalRng,-5,5);
			}
			
			model.setParameterVector(parameters);
			testWeightedDerivative(model, point, coefficients, 1.e-5,1.e-8);
		}
	}
	//test2: only second model
	{
		//parameters
		size_t modelParameters = 0;
		model.enableModelOptimization(0,false);
		model.enableModelOptimization(1,true);
		model.enableModelOptimization(2,false);
		BOOST_REQUIRE_EQUAL(model.numberOfParameters(),modelParameters);
		RealVector parameters(modelParameters);
		RealVector coefficients(2);
		RealVector point(3);
		for(unsigned int test = 0; test != 10; ++test){
			for(size_t i = 0; i != modelParameters;++i){
				parameters(i) = random::uni(random::globalRng,-5,5);
			}
			for(size_t i = 0; i != 2;++i){
				coefficients(i) = random::uni(random::globalRng,-5,5);
			}
			for(size_t i = 0; i != 3;++i){
				point(i) = random::uni(random::globalRng,-5,5);
			}
			
			model.setParameterVector(parameters);
			testWeightedDerivative(model, point, coefficients, 1.e-5,1.e-8);
		}
	}
	
	//test3: only third model
	{
		//parameters
		size_t modelParameters = net3.numberOfParameters();
		model.enableModelOptimization(0,false);
		model.enableModelOptimization(1,false);
		model.enableModelOptimization(2,true);
		BOOST_REQUIRE_EQUAL(model.numberOfParameters(),modelParameters);
		RealVector parameters(modelParameters);
		RealVector coefficients(2);
		RealVector point(3);
		for(unsigned int test = 0; test != 10; ++test){
			for(size_t i = 0; i != modelParameters;++i){
				parameters(i) = random::uni(random::globalRng,-5,5);
			}
			for(size_t i = 0; i != 2;++i){
				coefficients(i) = random::uni(random::globalRng,-5,5);
			}
			for(size_t i = 0; i != 3;++i){
				point(i) = random::uni(random::globalRng,-5,5);
			}
			
			model.setParameterVector(parameters);
			testWeightedDerivative(model, point, coefficients, 1.e-5,1.e-8);
		}
	}
	
	//test4: no parameters
	{
		//parameters
		model.enableModelOptimization(0,false);
		model.enableModelOptimization(1,false);
		model.enableModelOptimization(2,false);
		BOOST_REQUIRE_EQUAL(model.numberOfParameters(), 0);
	}
}
BOOST_AUTO_TEST_CASE( CONCATENATED_MODEL_weightedInputDerivative )
{
	LinearModel<> net1;
	NeuronLayer<SoftmaxNeuron<> > net2(5);
	LinearModel<> net3;
	net1.setStructure(3,5,1);
	net3.setStructure(5,10,1);
	ConcatenatedModel<RealVector> model  = net1 >> net2 >> net3;

	size_t modelParameters = net1.numberOfParameters()+net3.numberOfParameters();
	RealVector parameters(modelParameters);
	RealVector coefficients(10);
	RealVector point(3);
	for(unsigned int test = 0; test != 100; ++test){
		for(size_t i = 0; i != modelParameters;++i){
			parameters(i) = random::uni(random::globalRng,-10,10);
		}
		for(size_t i = 0; i != 10;++i){
			coefficients(i) = random::uni(random::globalRng,-10,10);
		}
		for(size_t i = 0; i != 3;++i){
			point(i) = random::uni(random::globalRng,-10,10);
		}
		
		model.setParameterVector(parameters);
		testWeightedInputDerivative(model, point, coefficients, 1.e-5,1.e-5);
	}

}

BOOST_AUTO_TEST_CASE( CONCATENATED_MODEL_weightedDerivatives )
{
	LinearModel<> net1;
	NeuronLayer<SoftmaxNeuron<> > net2(5);
	LinearModel<> net3;
	net1.setStructure(3,5,1);
	net3.setStructure(5,10,1);
	ConcatenatedModel<RealVector> model  = net1 >> net2 >> net3;

	size_t modelParameters = net1.numberOfParameters()+net3.numberOfParameters();
	RealVector parameters(modelParameters);
	RealMatrix coeffBatch(5,10);
	RealMatrix pointBatch(5,3);
	for(unsigned int test = 0; test != 100; ++test){
		for(size_t i = 0; i != modelParameters;++i){
			parameters(i) = random::uni(random::globalRng,-10,10);
		}
		for(std::size_t k = 0; k != 5; ++k){
			for(size_t i = 0; i != 10;++i){
				coeffBatch(k,i) = random::uni(random::globalRng,-10,10);
			}
			for(size_t i = 0; i != 3;++i){
				pointBatch(k,i) = random::uni(random::globalRng,-10,10);
			}
		}
		
		model.setParameterVector(parameters);
		boost::shared_ptr<State> state = model.createState();
		RealMatrix output; 
		model.eval(pointBatch,output,*state);
		
		RealMatrix inputDerivative;
		RealVector parameterDerivative;
		model.weightedInputDerivative(pointBatch, output, coeffBatch,*state,inputDerivative);
		model.weightedParameterDerivative(pointBatch, output, coeffBatch,*state, parameterDerivative);
		RealMatrix testInputDerivative;
		RealVector testParameterDerivative;
		model.weightedDerivatives(pointBatch, output, coeffBatch,*state, testParameterDerivative, testInputDerivative);
		double errorInput = max(inputDerivative-testInputDerivative); 
		double errorParameter = max(parameterDerivative-testParameterDerivative); 
		
		BOOST_CHECK_SMALL(errorInput,1.e-10);
		BOOST_CHECK_SMALL(errorParameter,1.e-10);
	}

}

BOOST_AUTO_TEST_CASE( CONCATENATED_MODEL_SERIALIZE )
{
	LinearModel<> net1;
	NeuronLayer<SoftmaxNeuron<> > net2(5);
	LinearModel<> net3;
	net1.setStructure(10,5,1);
	net3.setStructure(5,10,1);
	ConcatenatedModel<RealVector> model  = net1 >> net2 >> net3;

	//parameters
	RealVector testParameters(model.numberOfParameters());
	for(size_t i=0;i!=model.numberOfParameters();++i){
		testParameters(i)=random::uni(random::globalRng,-1,1);
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
			input(j)=random::uni(random::globalRng,-1,1);
		}
		data.push_back(input);
		target.push_back(model(input));
	}
	RegressionDataset dataset = createLabeledDataFromRange(data,target);

	//now we serialize the model
	ostringstream outputStream;  
	TextOutArchive oa(outputStream);  
	oa << model;

	//and create a new model from the serialization
	LinearModel<> netTest1;
	NeuronLayer<SoftmaxNeuron<> > netTest2;
	LinearModel<> netTest3;
	ConcatenatedModel<RealVector> modelDeserialized  = netTest1 >> netTest2 >> netTest3;
	istringstream inputStream(outputStream.str());  
	TextInArchive ia(inputStream);
	ia >> modelDeserialized;
	
	//test whether serialization works
	//first simple parameter and topology check
	BOOST_CHECK_SMALL(norm_2(modelDeserialized.parameterVector() - testParameters),1.e-50);
	BOOST_REQUIRE_EQUAL(net1.inputShape(),netTest1.inputShape());
	BOOST_REQUIRE_EQUAL(net1.outputShape(),netTest1.outputShape());
	BOOST_REQUIRE_EQUAL(net2.inputShape(),netTest2.inputShape());
	BOOST_REQUIRE_EQUAL(net2.outputShape(),netTest2.outputShape());
	BOOST_REQUIRE_EQUAL(net3.inputShape(),netTest3.inputShape());
	BOOST_REQUIRE_EQUAL(net3.outputShape(),netTest3.outputShape());
	for (size_t i=0; i<1000; i++)
	{
		RealVector output = modelDeserialized(dataset.element(i).input);
		BOOST_CHECK_SMALL(norm_2(output -dataset.element(i).label),1.e-8);
	}
}

BOOST_AUTO_TEST_SUITE_END()
