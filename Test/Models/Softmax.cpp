#define BOOST_TEST_MODULE ML_Softmax
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Models/Softmax.h>
#include "derivativeTestHelper.h"
#include <cmath>
#include <sstream>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/polymorphic_text_oarchive.hpp>

using namespace std;
using namespace boost::archive;
using namespace shark;

BOOST_AUTO_TEST_CASE( Softmax_Value )
{
	Softmax model(2);
	
	BOOST_CHECK_EQUAL(model.numberOfParameters(),0u);
	BOOST_CHECK_EQUAL(model.inputSize(),2u);
	BOOST_CHECK_EQUAL(model.outputSize(),2u);

	//the testpoint
	RealVector point(2);
	point(0)=1;
	point(1)=3;

	RealVector testResult(2);
	testResult(0)=exp(1.0);
	testResult(1)=exp(3.0);
	double sum=testResult(0)+testResult(1);
	testResult/=sum;


	//evaluate point
	RealVector result=model(point);
	double difference=norm_sqr(testResult-result);
	BOOST_CHECK_SMALL(difference,1.e-15);
}

//test whether the special case of single input is the same as dualinput with inputs (x,-x)
BOOST_AUTO_TEST_CASE( Softmax_Value_Single )
{
	Softmax model(1);
	Softmax modelTest(2);
	
	BOOST_CHECK_EQUAL(model.numberOfParameters(),0u);
	BOOST_CHECK_EQUAL(model.inputSize(),1u);
	BOOST_CHECK_EQUAL(model.outputSize(),2u);

	//the testpoint
	RealVector point(1);
	point(0)=1;
	RealVector pointTest(2);
	pointTest(0)=1;
	pointTest(1)=-1;


	//evaluate point
	RealVector result=model(point);
	RealVector resultTest=modelTest(pointTest);
	std::cout<<result<<resultTest<<std::endl;
	double difference=norm_sqr(resultTest-result);
	BOOST_CHECK_SMALL(difference,1.e-15);
}

BOOST_AUTO_TEST_CASE( Softmax_weightedParameterDerivative )
{
	Softmax model(2);

	testWeightedDerivative(model);
}
BOOST_AUTO_TEST_CASE( Softmax_weightedInputDerivative )
{
	{
		Softmax model(2);

		testWeightedDerivative(model);
	}
	{
		Softmax model(1);

		testWeightedDerivative(model);
	}
}

BOOST_AUTO_TEST_CASE( Softmax_SERIALIZE )
{
	//the target modelwork
	Softmax model(5);

	//create random parameters
	RealVector testParameters(model.numberOfParameters());
	for(size_t param=0;param!=model.numberOfParameters();++param)
	{
		 testParameters(param)=Rng::gauss(0,1);
	}
	model.setParameterVector( testParameters );

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
	RegressionDataset dataset = createLabeledDataFromRange(data,target);

	//now we serialize the FFmodel
	
	ostringstream outputStream;  
	polymorphic_text_oarchive oa(outputStream);  
	oa << const_cast<const Softmax&>(model);

	//and create a new model from the serialization
	Softmax modelDeserialized;
	istringstream inputStream(outputStream.str());  
	polymorphic_text_iarchive ia(inputStream);
	ia >> modelDeserialized;
	
	//test whether serialization works
	//first simple parameter and topology check
	BOOST_CHECK_SMALL(norm_2(modelDeserialized.parameterVector() - testParameters),1.e-50);
	BOOST_REQUIRE_EQUAL(modelDeserialized.inputSize(),model.inputSize());
	BOOST_REQUIRE_EQUAL(modelDeserialized.outputSize(),model.outputSize());
	for (size_t i=0; i<1000; i++)
	{
		RealVector output = modelDeserialized(dataset.element(i).input);
		BOOST_CHECK_SMALL(norm_2(output -dataset.element(i).label),1.e-50);
	}
}
