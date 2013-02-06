#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#define BOOST_TEST_MODULE ML_FFNET
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <shark/Models/FFNet.h>
#include <cmath>
#include <sstream>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/polymorphic_text_oarchive.hpp>
#include <shark/Rng/GlobalRng.h>

using namespace std;
using namespace boost::archive;
using namespace shark;

//this is the construction for a network with two hidden layers and bias as constructed by setStructure
bool fullConnections[6][7]=
{
	{0,0,0,0,0,0,0},//First Input
	{0,0,0,0,0,0,0},//second Input-both should be empty
	{1,1,0,0,0,0,1},//First Hidden
	{1,1,1,0,0,0,1},//Second Hidden - depends on first hidden
	{1,1,1,1,0,0,1},//First Output
	{1,1,1,1,0,0,1},//Second Output
};
//weights per neuron
int fullLayer1[] ={0,1};
int fullLayer2[] ={2,3,4};
int fullLayer3[] ={5,6,7,8};
int fullLayer4[] ={9,10,11,12};

int customMatrix[6][7]=
{
	{0,0,0,0,0,0,0},//First Input
	{0,0,0,0,0,0,0},//second Input-both should be empty
	{1,1,0,0,0,0,1},//First Hidden
	{0,0,1,0,0,0,1},//Second Hidden
	{0,1,1,0,0,0,0},//First Output
	{0,0,0,1,1,0,1},//Second Output
};


const size_t fullNumberOfWeights=17;
const size_t customNumberOfWeights=10;

double activation(double a)
{
	return 1.0/(1.0+std::exp(-a));
}

BOOST_AUTO_TEST_CASE( FFNET_setStructure )
{
	//2 input 2 output
	FFNet<LogisticNeuron,LogisticNeuron> net;

	//copy fullConnection Array into matrix
	std::vector<size_t> layer;
	layer.push_back(2);
	layer.push_back(1);
	layer.push_back(1);
	layer.push_back(2);

	//create a fully connected network
	net.setStructure(layer,true,true,true,true);
	//check connection matrix
	for(size_t i=0;i!=6;++i){
		for(size_t j=0;j!=7;++j){
			BOOST_CHECK_EQUAL(net.connections()(i,j) > 0,fullConnections[i][j]);
		}
	}
	//check layer structure
	BOOST_REQUIRE_EQUAL(net.layerMatrices().size(),3u);//3 layers
	BOOST_CHECK_EQUAL(net.layerMatrices()[0].size1(), 1u);//1 neuron in layer 1
	BOOST_CHECK_EQUAL(net.layerMatrices()[0].size2(), 2u);//2 connections in layer 1
	BOOST_CHECK_EQUAL(net.layerMatrices()[1].size1(), 1u);//1 neuron in layer 2
	BOOST_CHECK_EQUAL(net.layerMatrices()[1].size2(), 3u);//3 connections in layer 2
	BOOST_CHECK_EQUAL(net.layerMatrices()[2].size1(), 2u);//2 neuron in layer 3
	BOOST_CHECK_EQUAL(net.layerMatrices()[2].size2(), 4u);//4 connections in layer 3
	//check backprop structure
	BOOST_REQUIRE_EQUAL(net.backpropMatrices().size(), 3u);//3 layers
	BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size1(), 2u);//2 neuron in layer 1
	BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size2(), 4u);//4 connections in layer 1
	BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size1(), 1u);//1 neuron in layer 2
	BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size2(), 3u);//3 connections in layer 2
	BOOST_CHECK_EQUAL(net.backpropMatrices()[2].size1(), 1u);//1 neuron in layer 3
	BOOST_CHECK_EQUAL(net.backpropMatrices()[2].size2(), 2u);//2 connections in layer 3
	
	//now test parameter setting/getting
	RealVector params = net.parameterVector();
	BOOST_REQUIRE_EQUAL(net.numberOfParameters(),fullNumberOfWeights);
	BOOST_REQUIRE_EQUAL(params.size(),fullNumberOfWeights);
	
	for(std::size_t i = 0; i != 20; ++i){
		for(std::size_t j = 0; j != fullNumberOfWeights; ++j){
			params(j) = Rng::uni(0,1);
		}
		net.setParameterVector(params);
		RealVector paramResult = net.parameterVector();
		BOOST_CHECK_SMALL(normSqr(params-paramResult),1.e-25);
		
		//now also test, whether the weights are at the correct positions of the layer matrices
		BOOST_CHECK_SMALL(net.layerMatrices()[0](0,0)-params(0), 1.e-25);
		BOOST_CHECK_SMALL(net.layerMatrices()[0](0,1)-params(1), 1.e-25);
		BOOST_CHECK_SMALL(net.layerMatrices()[1](0,0)-params(2), 1.e-25);
		BOOST_CHECK_SMALL(net.layerMatrices()[1](0,1)-params(3), 1.e-25);
		BOOST_CHECK_SMALL(net.layerMatrices()[1](0,2)-params(4), 1.e-25);
		for(std::size_t i = 0; i != 2; ++i){
			for(std::size_t j = 0; j != 4; ++j){
				BOOST_CHECK_SMALL(net.layerMatrices()[2](i,j)-params(5+i*4+j), 1.e-25);
			}
		}
		
		//now also test, whether the weights are at the correct positions of the backprop matrices
		BOOST_CHECK_SMALL(net.backpropMatrices()[0](0,0)-params(0), 1.e-25);
		BOOST_CHECK_SMALL(net.backpropMatrices()[0](1,0)-params(1), 1.e-25);
		BOOST_CHECK_SMALL(net.backpropMatrices()[0](0,1)-params(2), 1.e-25);
		BOOST_CHECK_SMALL(net.backpropMatrices()[0](1,1)-params(3), 1.e-25);
		BOOST_CHECK_SMALL(net.backpropMatrices()[0](0,2)-params(5), 1.e-25);
		BOOST_CHECK_SMALL(net.backpropMatrices()[0](1,2)-params(6), 1.e-25);
		BOOST_CHECK_SMALL(net.backpropMatrices()[0](0,3)-params(9), 1.e-25);
		BOOST_CHECK_SMALL(net.backpropMatrices()[0](1,3)-params(10), 1.e-25);
		BOOST_CHECK_SMALL(net.backpropMatrices()[1](0,0)-params(4), 1.e-25);
		BOOST_CHECK_SMALL(net.backpropMatrices()[1](0,1)-params(7), 1.e-25);
		BOOST_CHECK_SMALL(net.backpropMatrices()[1](0,2)-params(11), 1.e-25);
		BOOST_CHECK_SMALL(net.backpropMatrices()[2](0,0)-params(8), 1.e-25);
		BOOST_CHECK_SMALL(net.backpropMatrices()[2](0,1)-params(12), 1.e-25);
		
		
		
	}
}
BOOST_AUTO_TEST_CASE( FFNET_Structure_customMatrix )
{
	FFNet<LogisticNeuron,LogisticNeuron> net;

	//copy Array into matrix
	IntMatrix cmat(6,7);
	for(size_t i=0;i!=6;++i){
		for(size_t j=0;j!=7;++j){
			cmat(i,j) = customMatrix[i][j];
		}
	}
	//2 input 1 output
	net.setStructure(2,1,cmat);
	
	//check layer structure
	BOOST_REQUIRE_EQUAL(net.layerMatrices().size(),3u);//3 layers
	BOOST_CHECK_EQUAL(net.layerMatrices()[0].size1(), 1u);//1 neuron in layer 1
	BOOST_CHECK_EQUAL(net.layerMatrices()[0].size2(), 2u);//2 connections in layer 1
	BOOST_CHECK_EQUAL(net.layerMatrices()[1].size1(), 2u);//2 neuron in layer 2
	BOOST_CHECK_EQUAL(net.layerMatrices()[1].size2(), 2u);//2 connections in layer 2
	BOOST_CHECK_EQUAL(net.layerMatrices()[2].size1(), 1u);//1 neuron in layer 3
	BOOST_CHECK_EQUAL(net.layerMatrices()[2].size2(), 2u);//2 connections in layer 3
	//check backprop structure
	BOOST_REQUIRE_EQUAL(net.backpropMatrices().size(), 3u);//3 layers
	BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size1(), 2u);//2 neuron in layer 1
	BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size2(), 3u);//3 connections in layer 1
	BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size1(), 1u);//1 neuron in layer 2
	BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size2(), 2u);//2 connections in layer 2
	BOOST_CHECK_EQUAL(net.backpropMatrices()[2].size1(), 2u);//2 neuron in layer 3
	BOOST_CHECK_EQUAL(net.backpropMatrices()[2].size2(), 1u);//1 connections in layer 3
	
	//now test parameter setting/getting
	RealVector params = net.parameterVector();
	BOOST_REQUIRE_EQUAL(net.numberOfParameters(),customNumberOfWeights);
	BOOST_REQUIRE_EQUAL(params.size(),customNumberOfWeights);
	
	for(std::size_t i = 0; i != 20; ++i){
		for(std::size_t j = 0; j != customNumberOfWeights; ++j){
			params(j) = Rng::uni(0,1);
		}
		net.setParameterVector(params);
		RealVector paramResult = net.parameterVector();
		BOOST_CHECK_SMALL(normSqr(params-paramResult),1.e-25);
		
		BOOST_CHECK_SMALL(net.layerMatrices()[0](0,0)-params(0), 1.e-25);
		BOOST_CHECK_SMALL(net.layerMatrices()[0](0,1)-params(1), 1.e-25);
		BOOST_CHECK_SMALL(net.layerMatrices()[1](0,0), 1.e-25);
		BOOST_CHECK_SMALL(net.layerMatrices()[1](0,1)- params(2), 1.e-25);
		BOOST_CHECK_SMALL(net.layerMatrices()[1](1,0)- params(3), 1.e-25);
		BOOST_CHECK_SMALL(net.layerMatrices()[1](1,1)- params(4), 1.e-25);
		BOOST_CHECK_SMALL(net.layerMatrices()[2](0,0)- params(5), 1.e-25);
		BOOST_CHECK_SMALL(net.layerMatrices()[2](0,1)- params(6), 1.e-25);
	}
}
BOOST_AUTO_TEST_CASE( FFNET_Value )
{
	//2 input 2 output
	FFNet<LogisticNeuron,LogisticNeuron> net;

	//copy fullConnection Array into matrix
	IntMatrix connections(6,7);
	for(size_t i=0;i!=6;++i)
		for(size_t j=0;j!=7;++j)
			connections(i,j)=fullConnections[i][j];
	net.setStructure(2,2,connections);

	for(std::size_t i = 0; i != 100; ++i){
		//initialize parameters
		RealVector parameters(fullNumberOfWeights);
		for(size_t j=0;j!=fullNumberOfWeights;++j)
			parameters(j)=Rng::gauss(0,1);
		net.setParameterVector(parameters);

		//the testpoint
		RealVector point(2);
		point(0)=Rng::uni(-5,5);
		point(1)= Rng::uni(-5,5);

		//evaluate ground truth result
		RealVector testActivations(6);
		testActivations(0)=point(0);
		testActivations(1)=point(1);
		double input2 = testActivations(0)*parameters(0)+testActivations(1)*parameters(1);
		testActivations(2)=activation(input2 + parameters(13));
		double input3 = testActivations(0)*parameters(2)+testActivations(1)*parameters(3)+ testActivations(2)*parameters(4);
		testActivations(3)=activation(input3 + parameters(14));
		double input4 = testActivations(0)*parameters(5)+testActivations(1)*parameters(6)+ testActivations(2)*parameters(7);
		input4 += testActivations(3)*parameters(8);
		testActivations(4)=activation(input4 + parameters(15));
		double input5 = testActivations(0)*parameters(9)+testActivations(1)*parameters(10)+ testActivations(2)*parameters(11);
		input5 += testActivations(3)*parameters(12);
		testActivations(5)=activation(input5 + parameters(16));
		
		//check whether final result is correct
		RealVector netResult = net(point);
		BOOST_CHECK_SMALL(netResult(0)-testActivations(4),1.e-12);
		BOOST_CHECK_SMALL(netResult(1)-testActivations(5),1.e-12);
		
		//now do the same for eval with batch and state
		RealMatrix batchPoints(1,2);
		row(batchPoints,0)=point;
		RealMatrix netResultBatch;
		boost::shared_ptr<State> state= net.createState();
		RealMatrix netResultBatchState;
		
		net.eval(batchPoints,netResultBatch);
		net.eval(batchPoints,netResultBatchState,*state);
		
		BOOST_REQUIRE_EQUAL(netResultBatch.size1(),1u);
		BOOST_REQUIRE_EQUAL(netResultBatch.size2(),2u);
		BOOST_REQUIRE_EQUAL(netResultBatchState.size1(),1u);
		BOOST_REQUIRE_EQUAL(netResultBatchState.size2(),2u);
		
		BOOST_CHECK_SMALL(netResultBatch(0,0)-testActivations(4),1.e-12);
		BOOST_CHECK_SMALL(netResultBatch(0,1)-testActivations(5),1.e-12);
		BOOST_CHECK_SMALL(netResultBatchState(0,0)-testActivations(4),1.e-12);
		BOOST_CHECK_SMALL(netResultBatchState(0,1)-testActivations(5),1.e-12);
		
		for(size_t i=0;i!=6;++i){
			BOOST_CHECK_SMALL(testActivations(i)-net.neuronResponses(*state)(i,0),1.e-15);
		}
	}
	
}
BOOST_AUTO_TEST_CASE( FFNET_Value_Custom )
{
	//2 input 2 output
	FFNet<LogisticNeuron,LogisticNeuron> net;

	//copy fullConnection Array into matrix
	IntMatrix connections(6,7);
	for(size_t i=0;i!=6;++i)
		for(size_t j=0;j!=7;++j)
			connections(i,j)=customMatrix[i][j];
	net.setStructure(2,1,connections);


	for(std::size_t i = 0; i != 100; ++i){
		//initialize parameters
		RealVector parameters(fullNumberOfWeights);
		for(size_t j=0;j!=fullNumberOfWeights;++j)
			parameters(j)=Rng::gauss(0,1);
		net.setParameterVector(parameters);

		//the testpoint
		RealVector point(2);
		point(0)=Rng::uni(-5,5);
		point(1)= Rng::uni(-5,5); 

		RealVector testActivations(6);
		testActivations(0)=point(0);
		testActivations(1)=point(1);
		double input2 = testActivations(0)*parameters(0)+testActivations(1)*parameters(1);
		testActivations(2)=activation(input2 + parameters(7));
		double input3 = testActivations(2)*parameters(2);
		testActivations(3)=activation(input3 + parameters(8));
		double input4 = testActivations(1)*parameters(3) + testActivations(2)*parameters(4);
		testActivations(4)=activation(input4);
		double input5 = testActivations(3)*parameters(5)+ testActivations(4)*parameters(6);
		testActivations(5)=activation(input5 + parameters(9)); 

		//check whether final result is correct
		RealVector netResult = net(point);
		BOOST_CHECK_SMALL(netResult(0)-testActivations(5),1.e-12);
		
		//now do the same for eval with batch and state
		RealMatrix batchPoints(1,2);
		row(batchPoints,0)=point;
		RealMatrix netResultBatch;
		boost::shared_ptr<State> state= net.createState();
		RealMatrix netResultBatchState;
		
		net.eval(batchPoints,netResultBatch);
		net.eval(batchPoints,netResultBatchState,*state);
		
		BOOST_REQUIRE_EQUAL(netResultBatch.size1(),1u);
		BOOST_REQUIRE_EQUAL(netResultBatch.size2(),1u);
		BOOST_REQUIRE_EQUAL(netResultBatchState.size1(),1u);
		BOOST_REQUIRE_EQUAL(netResultBatchState.size2(),1u);
		
		BOOST_CHECK_SMALL(netResultBatch(0,0)-testActivations(5),1.e-12);
		BOOST_CHECK_SMALL(netResultBatchState(0,0)-testActivations(5),1.e-12);
		
		for(size_t i=0;i!=6;++i){
			BOOST_CHECK_SMALL(testActivations(i)-net.neuronResponses(*state)(i,0),1.e-15);
		}
	}
}
BOOST_AUTO_TEST_CASE( FFNET_WeightedDerivative )
{
	//2 input 2 output
	FFNet<LogisticNeuron,LogisticNeuron> net;
	IntMatrix connections(6,7);
	//copy fullConnection Array into matrix
	for(size_t i=0;i!=6;++i)
		for(size_t j=0;j!=7;++j)
			connections(i,j)=fullConnections[i][j];

	//first check whether the number of weights is correct
	net.setStructure(2,2,connections);

	testWeightedDerivative(net,10000,5.e-5,1.e-7);
}
BOOST_AUTO_TEST_CASE( FFNET_WeightedDerivative_Custom )
{
	//2 input 2 output
	FFNet<LogisticNeuron,LogisticNeuron> net;
	IntMatrix connections(6,7);
	//copy fullConnection Array into matrix
	for(size_t i=0;i!=6;++i)
		for(size_t j=0;j!=7;++j)
			connections(i,j)=customMatrix[i][j];

	//first check whether the number of weights is correct
	net.setStructure(2,1,connections);

	testWeightedDerivative(net,10000,5.e-6,1.e-7);
}
BOOST_AUTO_TEST_CASE( FFNET_SERIALIZE )
{
	//the target network
	FFNet<LogisticNeuron,LogisticNeuron> net;
	IntMatrix connections(6,7);
	//copy fullConnection Array into matrix
	for(size_t i=0;i!=6;++i)
		for(size_t j=0;j!=7;++j)
			connections(i,j)=fullConnections[i][j];
	net.setStructure(2,2,connections);
	//create random parameters
	RealVector testParameters(net.numberOfParameters());
	for(size_t param=0;param!=net.numberOfParameters();++param)
	{
		 testParameters(param)=Rng::gauss(0,1);
	}
	net.setParameterVector( testParameters);

	//the test is, that after deserialization, the results must be identical
	//so we generate some data first
	std::vector<RealVector> data;
	std::vector<RealVector> target;
	RealVector input(net.inputSize());
	RealVector output(net.outputSize());
	for (size_t i=0; i<1000; i++)
	{
		for(size_t j=0;j!=net.inputSize();++j)
		{
			input(j)=Rng::uni(-1,1);
		}
		data.push_back(input);
		target.push_back(net(input));
	}
	RegressionDataset dataset(data,target);

	//now we serialize the FFNet
	ostringstream outputStream;  
	polymorphic_text_oarchive oa(outputStream);  
	oa << net;

	//and create a new net from the serialization
	FFNet<LogisticNeuron,LogisticNeuron> netDeserialized;
	istringstream inputStream(outputStream.str());  
	polymorphic_text_iarchive ia(inputStream);
	ia >> netDeserialized;
	
	//test whether serialization works
	//first simple parameter and topology check
	BOOST_CHECK_SMALL(norm_2(netDeserialized.parameterVector() - testParameters),1.e-50);
	BOOST_REQUIRE_EQUAL(netDeserialized.inputSize(),net.inputSize());
	BOOST_REQUIRE_EQUAL(netDeserialized.outputSize(),net.outputSize());
	for(size_t i=0;i!=6;++i)
		for(size_t j=0;j!=7;++j)
			BOOST_REQUIRE_EQUAL(netDeserialized.connections()(i,j),net.connections()(i,j));
	for (size_t i=0; i<1000; i++)
	{
		RealVector output = netDeserialized(dataset.element(i).input);
		BOOST_CHECK_SMALL(norm_2(output -dataset.element(i).label),1.e-2);
	}
}
