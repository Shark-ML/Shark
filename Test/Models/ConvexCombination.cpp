#include <shark/Models/ConvexCombination.h>
#include <shark/Models/LinearModel.h>

#define BOOST_TEST_MODULE Models_ConvexCombination
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <sstream>



using namespace std;
using namespace boost::archive;
using namespace shark;

//chck that parameters are set correctly
BOOST_AUTO_TEST_SUITE (Models_ConvexCombination)

BOOST_AUTO_TEST_CASE( Models_ConvexCombination_Parameters)
{
	std::size_t numTrials = 10;
	std::size_t inputDim = 10;
	std::size_t outputDim = 7;
	for(std::size_t t = 0; t != numTrials; ++t){
		//create the ground truth unnormalized and normalized weight matrices
		RealMatrix paramW(outputDim,inputDim);
		RealMatrix paramWUnnorm(outputDim,inputDim);
		for(std::size_t i = 0; i != outputDim; ++i){
			for(std::size_t j = 0; j != inputDim; ++j){
				paramWUnnorm(i,j) = std::abs(Rng::gauss(0,1));
				paramW(i,j) = paramWUnnorm(i,j);
			}
			row(paramW,i) /= sum(row(paramW,i));
		}
		
		RealVector paramsUnnorm = log(blas::adapt_vector(inputDim*outputDim,&paramWUnnorm(0,0)));
		RealVector params = log(blas::adapt_vector(inputDim*outputDim,&paramW(0,0)));
		
		//check normalized parameters
		{
			ConvexCombination model(inputDim,outputDim);
			model.setParameterVector(params);
			BOOST_REQUIRE_EQUAL(model.numberOfParameters(),inputDim * outputDim);
			
			//check that the constructed weight matrix is correct
			double error = norm_inf(paramW - model.weights());
			BOOST_CHECK_SMALL(error, 1.e-5);
			
			//check that the returned parameter vector is correct
			double errorParams = norm_inf(params - model.parameterVector());
			BOOST_CHECK_SMALL(errorParams, 1.e-5);
		}
		//check the unnormalized parameters
		{
			ConvexCombination model(inputDim,outputDim);
			model.setParameterVector(paramsUnnorm);
			BOOST_REQUIRE_EQUAL(model.numberOfParameters(),inputDim * outputDim);
			
			//check that the constructed weight matrix is correct
			double error = norm_inf(paramW - model.weights());
			BOOST_CHECK_SMALL(error, 1.e-5);
			
			//check that the returned parameter vector is correct
			double errorParams = norm_inf(params - model.parameterVector());
			BOOST_CHECK_SMALL(errorParams, 1.e-5);
		}
	}
}

//check that the model output is the same as the one of a linear model
BOOST_AUTO_TEST_CASE( Models_ConvexCombination_Value)
{
	std::size_t numTrials = 10;
	std::size_t inputDim = 10;
	std::size_t outputDim = 7;
	std::size_t numPoints = 100;
	for(std::size_t t = 0; t != numTrials; ++t){
		ConvexCombination model(inputDim,outputDim);
		BOOST_REQUIRE_EQUAL(model.numberOfParameters(),inputDim * outputDim);
		BOOST_REQUIRE_EQUAL(model.inputSize(),inputDim);
		BOOST_REQUIRE_EQUAL(model.outputSize(),outputDim);
		RealVector params(inputDim * outputDim);
		for(std::size_t i = 0; i != params.size(); ++i){
			params(i) = Rng::gauss(0,1);
		}
		model.setParameterVector(params);
		
		LinearModel<> modelTest(model.weights());
		RealMatrix inputs(numPoints,inputDim);
		for(std::size_t i = 0; i != numPoints; ++i){
			for(std::size_t j = 0; j != inputDim; ++j){
				inputs(i,j) = Rng::gauss(0,1);
			}
		}
		
		RealMatrix testOutput = modelTest(inputs);
		RealMatrix modelOutput = model(inputs);
		
		BOOST_REQUIRE_EQUAL(modelOutput.size1(), numPoints);
		BOOST_REQUIRE_EQUAL(modelOutput.size2(), outputDim);
		
		double error = norm_inf(testOutput - modelOutput);
		BOOST_CHECK_SMALL(error, 1.e-10);
	}
}

BOOST_AUTO_TEST_CASE( Models_ConvexCombination_Derivatives)
{
	ConvexCombination model(3, 5);
	BOOST_REQUIRE_EQUAL(model.numberOfParameters(), 15u);

	testWeightedDerivative(model,10000);
	testWeightedInputDerivative(model,10000);
}

BOOST_AUTO_TEST_CASE( Models_ConvexCombination_SERIALIZE )
{
	//the target modelwork
	ConvexCombination model(2, 3);

	//create random parameters
	RealVector testParameters(model.numberOfParameters());
	for(size_t param=0;param!=model.numberOfParameters();++param)
	{
		testParameters(param)=Rng::gauss(0,1);
	}
	model.setParameterVector( testParameters);
	testParameters = model.parameterVector(); //allow transformation of parameters
	//the test is, that after deserialization, the results must be identical
	//so we generate some data first
	std::vector<RealVector> data;
	std::vector<RealVector> target;
	RealVector input(model.inputSize());
	RealVector output(model.outputSize());
	for (size_t i=0; i<1000; i++)
	{
		for(size_t j=0;j!=model.inputSize();++j)
		{
			input(j)=Rng::uni(-1,1);
		}
		data.push_back(input);
		target.push_back(model(input));
	}
	
	RegressionDataset dataset  = createLabeledDataFromRange(data,target);

	//now we serialize the model
	
	ostringstream outputStream;  
	TextOutArchive oa(outputStream);  
	oa << model;
	//and create a new model from the serialization
	ConvexCombination modelDeserialized;
	istringstream inputStream(outputStream.str());  
	TextInArchive ia(inputStream);
	ia >> modelDeserialized;
	//test whether serialization works
	//first simple parameter and topology check
	BOOST_REQUIRE_EQUAL(modelDeserialized.numberOfParameters(),model.numberOfParameters());
	BOOST_CHECK_SMALL(norm_2(modelDeserialized.parameterVector() - testParameters),1.e-50);
	BOOST_REQUIRE_EQUAL(modelDeserialized.inputSize(),model.inputSize());
	BOOST_REQUIRE_EQUAL(modelDeserialized.outputSize(),model.outputSize());
	for (size_t i=0; i<1000; i++)
	{
		RealVector output = modelDeserialized(dataset.element(i).input);
		BOOST_CHECK_SMALL(norm_2(output -dataset.element(i).label),1.e-50);
	}
}

BOOST_AUTO_TEST_SUITE_END()
