#include <shark/Models/LinearModel.h>
#define BOOST_TEST_MODULE Models_LinearModel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <sstream>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/polymorphic_text_oarchive.hpp>

using namespace std;
using namespace boost::archive;
using namespace shark;


BOOST_AUTO_TEST_SUITE (Models_LinearModel)

BOOST_AUTO_TEST_CASE( Models_LinearModel )
{
	// 2 inputs, 2 outputs, no offset -> 4 parameters
	LinearModel<> model(2, 2, false);
	BOOST_CHECK_EQUAL(model.numberOfParameters(), 4u);

	// The matrix should have the form
	// [2 1]
	// [3 5]
	RealVector testParameters(4);
	testParameters(0) = 2;
	testParameters(1) = 1;
	testParameters(2) = 3;
	testParameters(3) = 5;

	// Test whether setting and getting of parameters works
	model.setParameterVector(testParameters);
	RealVector retrievedParameters = model.parameterVector();
	for (size_t i=0; i!=4; ++i)
		BOOST_CHECK_EQUAL(retrievedParameters(i), testParameters(i));

	// Test the evaluation function
	RealMatrix testInput(2,2);
	testInput(0,0) = 1;
	testInput(0,1) = 2;
	testInput(1,0) = 3;
	testInput(1,1) = 4;
	RealMatrix testResults(2,2);
	testResults(0,0) = 4;
	testResults(0,1) = 13;
	testResults(1,0) = 10;
	testResults(1,1) = 29;
	RealMatrix output=model(testInput);
	for (size_t i=0; i!=2; ++i)
		for (size_t j=0; j!=2; ++j)
			BOOST_CHECK_SMALL(output(i,j) - testResults(i,j), 10e-15);

	testWeightedDerivative(model,1000);
	testWeightedInputDerivative(model,1000);
}

BOOST_AUTO_TEST_CASE( Models_AffineLinearModel )
{
	// 2 inputs, 2 outputs, with offset -> 6 parameters
	LinearModel<> model(2, 2, true);
	BOOST_CHECK_EQUAL(model.numberOfParameters(), 6u);

	// matrix should have the form
	// [2 1]
	// [3 5]
	// offset should have the form
	// [1 -1]
	RealVector testParameters(6);
	testParameters(0) = 2;
	testParameters(1) = 1;
	testParameters(2) = 3;
	testParameters(3) = 5;
	testParameters(4) = 1;
	testParameters(5) = -1;

	// Test whether setting and getting of parameters works
	model.setParameterVector(testParameters);
	RealVector retrievedParameters=model.parameterVector();
	for(size_t i=0;i!=6;++i)
		BOOST_CHECK_EQUAL(retrievedParameters(i),testParameters(i));

	// Test the evaluation function
	RealMatrix testInput(2,2);
	testInput(0,0) = 1;
	testInput(0,1) = 2;
	testInput(1,0) = 3;
	testInput(1,1) = 4;
	RealMatrix testResults(2,2);
	testResults(0,0) = 5;
	testResults(0,1) = 12;
	testResults(1,0) = 11;
	testResults(1,1) = 28;
	RealMatrix output=model(testInput);
	for (size_t i=0; i!=2; ++i)
		for (size_t j=0; j!=2; ++j)
			BOOST_CHECK_SMALL(output(i,j) - testResults(i,j), 1.e-15);

	// Test the weighted derivatives
	testWeightedDerivative(model,1000);
	testWeightedInputDerivative(model,1000);
	//testWeightedSecondDerivative(model,testInput,coefficients,coeffHessians);
}

BOOST_AUTO_TEST_CASE( LinearModel_SERIALIZE )
{
	//the target modelwork
	LinearModel<> model(2, 2, true);

	//create random parameters
	RealVector testParameters(model.numberOfParameters());
	for(size_t param=0;param!=model.numberOfParameters();++param)
	{
		 testParameters(param)=Rng::gauss(0,1);
	}
	model.setParameterVector( testParameters);
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

	//now we serialize the FFmodel
	
	ostringstream outputStream;  
	polymorphic_text_oarchive oa(outputStream);  
	oa << model;
	//and create a new model from the serialization
	LinearModel<> modelDeserialized;
	istringstream inputStream(outputStream.str());  
	polymorphic_text_iarchive ia(inputStream);
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
