#define BOOST_TEST_MODULE ML_RNNET
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Models/RNNet.h>
#include <shark/Rng/GlobalRng.h>
#include <sstream>

using namespace shark;

int connections[6][9]=
{
	{1,1,1,1,1,1,1,1,1},
	{1,1,1,1,1,1,1,1,1},
	{1,1,1,1,1,1,1,1,1},
	{1,1,1,1,1,1,1,1,1},
	{1,1,1,1,1,1,1,1,1},
	{1,1,1,1,1,1,1,1,1}
};

const size_t numberOfParameters=54;

BOOST_AUTO_TEST_SUITE (Models_RNNet)

BOOST_AUTO_TEST_CASE( RNNET_SIMPLE_SET_STRUCTURE_TEST)
{
	RecurrentStructure netStruct;
	netStruct.setStructure(2,4,2);

	for (size_t i = 0; i < 6; i++){
		for (size_t j = 0; j < 9; j++){
			BOOST_CHECK_EQUAL(netStruct.connections()(i,j),connections[i][j]);
		}
	}
}

BOOST_AUTO_TEST_CASE( RNNET_SET_PARAMETER_TEST)
{
	RecurrentStructure netStruct;
	netStruct.setStructure(2,4,2);
	RNNet net(&netStruct);
	BOOST_REQUIRE_EQUAL(numberOfParameters, net.numberOfParameters());
	//create test parameters
	RealVector testParameters(numberOfParameters);
	for(size_t i=0;i!=numberOfParameters;++i){
		testParameters(i)=0.1*i;
	}

	//set parameters
	net.setParameterVector(testParameters);
	//test wether parameterVector works
	RealVector resultParameters = net.parameterVector();
	BOOST_CHECK_SMALL(norm_sqr(testParameters-resultParameters),1.e-30);

	//test wether the weight matrices are correct
	for (size_t i = 0; i < 6; i++){
		for (size_t j = 0; j < 9; j++){
			BOOST_CHECK_SMALL(netStruct.weights()(i,j)-(i*9+j)*0.1,1.e-10);
		}
	}
}

//this test compares the network to the MSEFFNET of Shark 2.4
//since the topology of the net changed, this is not that easy...
BOOST_AUTO_TEST_CASE( RNNET_BATCH_VALUE_REGRESSION_TEST ){
	//We simulate a Shark 2.4 network which would be created using
	//setStructure(2,2) and setting all feed forward connections to 0.
	//since the input units in the old network
	//where sigmoids, we have to use a hidden layer to emulate them
	//sicne this uses every feature of the topology possible, this
	//test should catch every possible error during eval.
	int regConnections[4][7]=
	{
		{1,0,0,0,0,0,0},//shark 2.4 input1
		{0,1,0,0,0,0,0},//shark 2.4 input 2
		{0,0,0,1,1,1,1},//output 1
		{0,0,0,1,1,1,1} //output 2
	};
	IntMatrix conn(4,7);
	for (size_t i = 0; i < 4; i++){
		for (size_t j = 0; j < 7; j++){
			conn(i,j)=regConnections[i][j];
		}
	}
	RecurrentStructure netStruct;
	netStruct.setStructure(2,2,conn);
	RNNet net(&netStruct);
	BOOST_REQUIRE_EQUAL(net.numberOfParameters(),10);


	//initialize parameters
	RealVector parameters(10);
	//our simulated input neurons need strength 1
	parameters(0)=1;
	parameters(1)=1;
	//rest is 0.1
	for(size_t i=2;i!=10;++i){
		parameters(i)=0.1*i-0.5;
	}
	net.setParameterVector(parameters);

	Sequence warmUp(1,RealVector(2));//just 1 element
	warmUp[0](0)=0;
	warmUp[0](1)=1;
	net.setWarmUpSequence(warmUp);

	//input and output data from an test of an earlier implementation
	Sequence testInputs(4,RealVector(2));
	for (size_t i = 0; i < 4; i++){
		for(size_t j=0;j!=2;++j){
			testInputs[i](j)  = i+j+1;
		}
	}
	Sequence testOutputs(4,RealVector(2));

	testOutputs[0](0)=0.414301;
	testOutputs[0](1)=0.633256;
	testOutputs[1](0)=0.392478;
	testOutputs[1](1)=0.651777;
	testOutputs[2](0)=0.378951;
	testOutputs[2](1)=0.658597;
	testOutputs[3](0)=0.372836;
	testOutputs[3](1)=0.661231;

	//eval network output and test wether it's the same or not
	Sequence outputs=net(testInputs);
	for(size_t i=0;i!=4;++i){
		BOOST_CHECK_SMALL(norm_2(outputs[i]-testOutputs[i]),1.e-5);
	}

	//in batch mode, a network should reset itself when a new batch is generated
	//check wether this happens - just repeat it
	outputs=net(testInputs);
	for(size_t i=0;i!=4;++i){
		BOOST_CHECK_SMALL(norm_2(outputs[i]-testOutputs[i]),1.e-5);
	}
}
BOOST_AUTO_TEST_CASE( RNNET_WEIGHTED_PARAMETER_DERIVATIVE ){
	const size_t T=5;

	RecurrentStructure netStruct;
	netStruct.setStructure(2,4,2);
	RNNet net(&netStruct);
	BOOST_REQUIRE_EQUAL(net.numberOfParameters(),numberOfParameters);

	//initialize parameters
	RealVector parameters(numberOfParameters);
	for(size_t i=0;i!=numberOfParameters;++i){
		parameters(i)= Rng::gauss(0,1);
	}
	net.setParameterVector(parameters);

	//define sequence
	Sequence testInputs(T,RealVector(2));
	for (size_t t = 0; t < T; t++){
		for(size_t j=0;j!=2;++j){
			testInputs[t](j)  = t+j;
		}
	}
	std::vector<Sequence> testInputBatch(1,testInputs);
	std::vector<Sequence> testOutputBatch;
	//we choose the same sequence as warmup sequence as to test the net
	net.setWarmUpSequence(testInputs);
	//evaluate network
	boost::shared_ptr<State> state = net.createState();
	
	net.eval(testInputBatch,testOutputBatch,*state);

	//define coefficients
	Sequence coefficients(T,RealVector(2));
	for (size_t t = 0; t < T; t++){
		for(size_t j=0;j!=2;++j){
			coefficients[t](j)  = 1;
		}
	}
	
	std::vector<Sequence> coefficientsBatch(1,coefficients);

	//now calculate the derivative
	RealVector derivative;
	net.weightedParameterDerivative(testInputBatch,coefficientsBatch,*state,derivative);
	BOOST_REQUIRE_EQUAL(derivative.size(),numberOfParameters);

	//estimate derivative.
	double epsilon=1.e-5;
	RealVector testDerivative(numberOfParameters);
	testDerivative.clear();
	for(size_t w=0; w != numberOfParameters; ++w){
		//create points with an change of +-epsilon in the wth component
		RealVector point1(parameters);
		RealVector point2(parameters);
		point1(w)+=epsilon;
		point2(w)-=epsilon;
		//calculate result
		net.setParameterVector(point1);
		Sequence result1=net(testInputs);
		net.setParameterVector(point2);
		Sequence result2=net(testInputs);

		//now estimate the derivative for the changed parameter
		for(size_t t=0;t!=T;++t){
			testDerivative(w)+=inner_prod(coefficients[t],(result1[t]-result2[t])/(2*epsilon));
		}

	}

	//check wether the derivatives are identical
	BOOST_CHECK_SMALL(blas::distance(derivative,testDerivative),epsilon);
	//~ for(size_t w=0; w != numberOfParameters; ++w){
		//~ std::cout<<derivative(w)<<" "<<testDerivative(w)<<std::endl;
	//~ }

}

//~ BOOST_AUTO_TEST_CASE( RNNET_SERIALIZATION_TEST)
//~ {
	//~ std::stringstream str;
	//~ TextOutArchive  archive(str);
	//~ RecurrentStructure netStruct;
	//~ netStruct.setStructure(2,4,2);
	//~ archive << netStruct;

//~ }

BOOST_AUTO_TEST_SUITE_END()
