#include <shark/Models/RBFLayer.h>

#define BOOST_TEST_MODULE ML_RBFLayer
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <sstream>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/polymorphic_text_oarchive.hpp>

using namespace std;
using namespace boost::archive;
using namespace shark;

BOOST_AUTO_TEST_SUITE (Models_RBFLayer)

BOOST_AUTO_TEST_CASE( RBFLayer_Parameters )
{
	//2 input 2 output
	RBFLayer net(2,3);

	{
		size_t numParams=net.numberOfParameters();
		BOOST_REQUIRE_EQUAL(numParams,9);
		RealVector parameters(numParams);
		for(size_t i=0;i!=numParams;i++)
			parameters(i)=i;
		net.setParameterVector(parameters);

		RealVector params = net.parameterVector();
		BOOST_REQUIRE_EQUAL(params.size(),numParams);
		double paramError = norm_inf(params-parameters);
		BOOST_CHECK_SMALL(paramError,1.e-14);
	}
	
	{
		net.setTrainingParameters(true,false);
		size_t numParams=net.numberOfParameters();
		BOOST_REQUIRE_EQUAL(numParams,6);
		RealVector parameters(numParams);
		for(size_t i=0;i!=numParams;i++)
			parameters(i)=i;
		net.setParameterVector(parameters);

		RealVector params = net.parameterVector();
		BOOST_REQUIRE_EQUAL(params.size(),numParams);
		double paramError = norm_inf(params-parameters);
		BOOST_CHECK_SMALL(paramError,1.e-14);
	}
	
	{
		net.setTrainingParameters(false,true);
		size_t numParams=net.numberOfParameters();
		BOOST_REQUIRE_EQUAL(numParams,3);
		RealVector parameters(numParams);
		for(size_t i=0;i!=numParams;i++)
			parameters(i)=i;
		net.setParameterVector(parameters);

		RealVector params = net.parameterVector();
		BOOST_REQUIRE_EQUAL(params.size(),numParams);
		double paramError = norm_inf(params-parameters);
		BOOST_CHECK_SMALL(paramError,1.e-14);
	}
	{
		net.setTrainingParameters(false,false);
		size_t numParams=net.numberOfParameters();
		BOOST_REQUIRE_EQUAL(numParams,0);
	}
	
	{
		net.setTrainingParameters(true,true);
		size_t numParams=net.numberOfParameters();
		BOOST_REQUIRE_EQUAL(numParams,9);
		RealVector parameters(numParams);
		for(size_t i=0;i!=numParams;i++)
			parameters(i)=i;
		net.setParameterVector(parameters);

		RealVector params = net.parameterVector();
		BOOST_REQUIRE_EQUAL(params.size(),numParams);
		double paramError = norm_inf(params-parameters);
		BOOST_CHECK_SMALL(paramError,1.e-14);
	}
	
}


BOOST_AUTO_TEST_CASE( RBFLayer_Value )
{
	//2 input 3 output
	RBFLayer net(2,3);

	//initialize parameters
	size_t numParams=net.numberOfParameters();
	BOOST_REQUIRE_EQUAL(numParams,9);

	RealVector parameters(numParams);
	for(size_t i=0;i!=numParams-3;i++)
		parameters(i)=(i+1)*0.5;
	parameters(6)= 1;
	parameters(7)= 0;
	parameters(8)= -1;
	net.setParameterVector(parameters);
	
	RealVector var(3);
	var(0) = 0.5/std::exp(1.0);
	var(1) = 0.5/std::exp(0.0);
	var(2) = 0.5/std::exp(-1.0);
	
	double pi = boost::math::constants::pi<double>();
	for(std::size_t i = 0; i != 1000; ++i){
		RealVector point(2);
		point(0)=Rng::gauss(0,2);
		point(1)=Rng::gauss(0,2);
		
		double dist0 = sqr(point(0)-0.5)+sqr(point(1)-1);
		double dist1 = sqr(point(0)-1.5)+sqr(point(1)-2);
		double dist2 = sqr(point(0)-2.5)+sqr(point(1)-3);
		
		double p0 = std::exp(-dist0/(2*var(0)))/(2*pi*var(0));
		double p1 = std::exp(-dist1/(2*var(1)))/(2*pi*var(1));
		double p2 = std::exp(-dist2/(2*var(2)))/(2*pi*var(2));

		RealVector result=net(point);
		BOOST_CHECK_SMALL(std::abs(result(0)-p0),1.e-13);
		BOOST_CHECK_SMALL(std::abs(result(1)-p1),1.e-13);
		BOOST_CHECK_SMALL(std::abs(result(2)-p2),1.e-13);
		
	}
	
	//batch eval
	RealMatrix inputBatch(100,2);
	for(std::size_t i = 0; i != 100; ++i){
		inputBatch(i,0) = Rng::uni(-1,1);
		inputBatch(i,1) = Rng::uni(-1,1);
	}
	testBatchEval(net,inputBatch);
}

BOOST_AUTO_TEST_CASE( RBFLayer_WeightedDerivative )
{
	//3 input, 5 output
	RBFLayer net(3,5);
	std::cout<<"v1"<<std::endl;
	net.setTrainingParameters(true,true);
	BOOST_REQUIRE_EQUAL(net.numberOfParameters(), 20);
	testWeightedDerivative(net,1000,1.e-6,1.e-7);
	std::cout<<"v2"<<std::endl;
	net.setTrainingParameters(true,false);
	BOOST_REQUIRE_EQUAL(net.numberOfParameters(), 15);
	testWeightedDerivative(net,1000,1.e-6,1.e-7);
	std::cout<<"v3"<<std::endl;
	net.setTrainingParameters(false,true);
	BOOST_REQUIRE_EQUAL(net.numberOfParameters(), 5);
	testWeightedDerivative(net,1000,1.e-6,1.e-7);
	std::cout<<"v4"<<std::endl;
	net.setTrainingParameters(false,false);
	BOOST_REQUIRE_EQUAL(net.numberOfParameters(), 0);
	testWeightedDerivative(net,1000,1.e-6,1.e-7);
}

BOOST_AUTO_TEST_CASE( RBFLayer_SERIALIZE )
{
	//the target modelwork
	RBFLayer model(2,3);

	//create random parameters
	RealVector testParameters(model.numberOfParameters());
	for(size_t param=0;param!=model.numberOfParameters();++param)
	{
		 testParameters(param) = Rng::uni(0,1);
	}
	model.setParameterVector( testParameters);
	model.setTrainingParameters(true,false);

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
	oa << model;

	//and create a new model from the serialization
	RBFLayer modelDeserialized;
	istringstream inputStream(outputStream.str());  
	polymorphic_text_iarchive ia(inputStream);
	ia >> modelDeserialized;
	
	//test whether serialization works
	//first simple parameter and topology check

	//serialization for restricted set to ensure, it is saved
	BOOST_REQUIRE_EQUAL(modelDeserialized.numberOfParameters(), model.numberOfParameters());
	BOOST_CHECK_SMALL(norm_2(modelDeserialized.parameterVector() - model.parameterVector()),1.e-15);
	modelDeserialized.setTrainingParameters(true,true);
	//full parameter check
	BOOST_CHECK_SMALL(norm_2(modelDeserialized.parameterVector() - testParameters),1.e-15);

	BOOST_REQUIRE_EQUAL(modelDeserialized.inputSize(),model.inputSize());
	BOOST_REQUIRE_EQUAL(modelDeserialized.outputSize(),model.outputSize());
	for (size_t i=0; i < 1000; i++)
	{
		RealVector output = modelDeserialized(dataset.element(i).input);
		BOOST_CHECK_SMALL(norm_2(output -dataset.element(i).label),1.e-50);
	}
}

BOOST_AUTO_TEST_SUITE_END()
