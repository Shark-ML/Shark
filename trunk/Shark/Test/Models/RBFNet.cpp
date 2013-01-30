#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#define BOOST_TEST_MODULE ML_RBFNet
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <shark/Models/RBFNet.h>
#include <cmath>

#include <sstream>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/polymorphic_text_oarchive.hpp>

using namespace std;
using namespace boost::archive;
using namespace shark;


BOOST_AUTO_TEST_CASE( RBFNet_Value )
{
	//2 input 2 output
	RBFNet net(2,3,2);

	//initialize parameters
	size_t numParams=net.numberOfParameters();
	BOOST_REQUIRE_EQUAL(numParams,17);

	RealVector parameters(numParams);
	for(size_t i=0;i!=numParams-3;i++)
		parameters(i)=0.5;
	parameters(14)=parameters(15)=parameters(16)=std::log(0.5);
	net.setParameterVector(parameters);

	//test wether the parameter vector is generated correctly
	RealVector params = net.parameterVector();
	BOOST_REQUIRE_EQUAL(params.size(),17);
	double paramError = norm_2(params-parameters);
	BOOST_CHECK_SMALL(paramError,1.e-15);

	//the testpoint
	RealVector point(2);
	point(0)=-0.5;
	point(1)=-0.5;

	//calculate result
	double hiddenResult = std::exp( -1. );
	RealVector testResult(2);
	testResult(0)=1.5*hiddenResult+0.5;
	testResult(1)=testResult(0);

	//evaluate point
	RealVector result=net(point);
	double evalError=norm_2(result-testResult);
	BOOST_CHECK_SMALL(evalError,1.e-15);
}

BOOST_AUTO_TEST_CASE( RBFNet_WeightedDerivative )
{
	//3 input, 5 hidden, 2 output
	RBFNet net(3,5,2);
	std::cout<<"v1"<<std::endl;
	net.setTrainingParameters(true,true,true);
	testWeightedDerivative(net,1000,5.e-5,1.e-7);
	std::cout<<"v2"<<std::endl;
	net.setTrainingParameters(true,true,false);
	testWeightedDerivative(net,1000,5.e-5,1.e-7);
	std::cout<<"v3"<<std::endl;
	net.setTrainingParameters(false,false,true);
	testWeightedDerivative(net,1000,5.e-5,1.e-7);
	std::cout<<"v4"<<std::endl;
	net.setTrainingParameters(false,true,false);
	testWeightedDerivative(net,1000,5.e-5,1.e-7);
	std::cout<<"v5"<<std::endl;
	net.setTrainingParameters(false,true,true);
	testWeightedDerivative(net,1000,5.e-5,1.e-7);
	std::cout<<"v6"<<std::endl;
	net.setTrainingParameters(true,false,false);
	testWeightedDerivative(net,1000,5.e-5,1.e-7);
	std::cout<<"v7"<<std::endl;
	net.setTrainingParameters(true,false,true);
	testWeightedDerivative(net,1000,5.e-5,1.e-7);
}

BOOST_AUTO_TEST_CASE( RBFNet_SERIALIZE )
{
	//the target modelwork
	RBFNet model(2,2,3);

	//create random parameters
	RealVector testParameters(model.numberOfParameters());
	for(size_t param=0;param!=model.numberOfParameters();++param)
	{
		 testParameters(param) = Rng::uni(0,1);
	}
	model.setParameterVector( testParameters);
	model.setTrainingParameters(true,false,false);

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
	RegressionDataset dataset(data,target);

	//now we serialize the FFmodel
	
	ostringstream outputStream;  
	polymorphic_text_oarchive oa(outputStream);  
	oa << model;

	//and create a new model from the serialization
	RBFNet modelDeserialized;
	istringstream inputStream(outputStream.str());  
	polymorphic_text_iarchive ia(inputStream);
	ia >> modelDeserialized;
	
	//test whether serialization works
	//first simple parameter and topology check

	//serialization for restricted set to ensure, it is saved
	BOOST_REQUIRE_EQUAL(modelDeserialized.numberOfParameters(), model.numberOfParameters());
	BOOST_CHECK_SMALL(norm_2(modelDeserialized.parameterVector() - model.parameterVector()),1.e-15);
	modelDeserialized.setTrainingParameters(true,true,true);
	//full parameter check
	BOOST_CHECK_SMALL(norm_2(modelDeserialized.parameterVector() - testParameters),1.e-15);

	BOOST_REQUIRE_EQUAL(modelDeserialized.inputSize(),model.inputSize());
	BOOST_REQUIRE_EQUAL(modelDeserialized.outputSize(),model.outputSize());
	for (size_t i=0; i < 1000; i++)
	{
		RealVector output = modelDeserialized(dataset(i).input);
		BOOST_CHECK_SMALL(norm_2(output -dataset(i).label),1.e-50);
	}
}
