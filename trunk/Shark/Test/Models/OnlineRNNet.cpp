#define BOOST_TEST_MODULE ML_RNNET
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Models/OnlineRNNet.h>
#include <shark/Rng/GlobalRng.h>
#include <sstream>

using namespace shark;


//this test compares the network to the MSEFFNET of Shark 2.4
//since the topology of the net changed, this is not that easy...
BOOST_AUTO_TEST_CASE( ONLINERNNET_VALUE_TEST ){
	//We simulate a Shark 2.4 network which would be created using
	//setStructure(2,2) and setting all feed forward connections to 0.
	//since the input units in the old network
	//where sigmoids, we have to use a hidden layer to emulate them
	//since this uses every feature of the topology possible, this
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
	OnlineRNNet net(&netStruct);
	BOOST_REQUIRE_EQUAL(net.numberOfParameters(),10);


	//initialize parameters
	RealVector parameters(10);
	//our simulated input neurons need strength 1
	parameters(0)=1; 
	parameters(1)=1;
	for(size_t i=2;i!=10;++i){
		parameters(i)=0.1*i-0.5;
	}
	net.setParameterVector(parameters);

	std::cout<<parameters<<std::endl;
	std::cout<<netStruct.weights()<<std::endl;
	//input and output data from an test of an earlier implementation
	RealMatrix testInputs(5,2);
	for (size_t i = 0; i < 5; i++){
		for(size_t j=0;j!=2;++j){
			testInputs(i,j)  = i+j;
		}
	}
	RealMatrix testOutputs(5,2);

	testOutputs(0,0)=0.5;
	testOutputs(0,1)=0.5;
	testOutputs(1,0)=0.414301;
	testOutputs(1,1)=0.633256;
	testOutputs(2,0)=0.392478;
	testOutputs(2,1)=0.651777;
	testOutputs(3,0)=0.378951;
	testOutputs(3,1)=0.658597;
	testOutputs(4,0)=0.372836;
	testOutputs(4,1)=0.661231;

	//eval network output and test wether it's the same or not
	for(size_t i=0;i!=5;++i){
		RealVector output=net(row(testInputs,i));
		std::cout<<output(0)<<" "<<output(1)<<std::endl;
		BOOST_CHECK_SMALL(norm_2(output-row(testOutputs,i)),1.e-5);
	}

	//now, after resetting the network, we should get the same result
	net.resetInternalState();
	for(size_t i=0;i!=5;++i){
		RealVector output=net(row(testInputs,i));
		BOOST_CHECK_SMALL(norm_2(output-row(testOutputs,i)),1.e-5);
	}
}
BOOST_AUTO_TEST_CASE( ONLINE_RNNET_WEIGHTED_PARAMETER_DERIVATIVE ){
	RecurrentStructure netStruct;
	netStruct.setStructure(2,4,2,true);
	OnlineRNNet net(&netStruct);
	const size_t T=10;
	const size_t numberOfParameters=54;
	BOOST_REQUIRE_EQUAL(net.numberOfParameters(),numberOfParameters);

	//initialize parameters
	RealVector parameters(numberOfParameters);
	for(size_t i=0;i!=numberOfParameters;++i){
		parameters(i)= Rng::gauss(0,1)-0.1;
	}
	net.setParameterVector(parameters);

	//define sequence
	RealMatrix testInputs(T,2);
	for (size_t t = 0; t < T; t++){
		for(size_t j=0;j!=2;++j){
			testInputs(t,j)  = Rng::uni(-1,1);
		}
	}
	//define coefficients
	RealMatrix coefficients(1,2);
	coefficients(0,0)  = 0.5;
	coefficients(0,1)  = 1;

	//we test the derivative for every subsequence [0,t]
	for(size_t t=0;t != T; ++t){
		net.resetInternalState();
		//run subsequence the first time and calculate iterative derivative
		RealVector derivative;
		for(size_t t2=0;t2 <=t; ++t2){
			RealMatrix input(1,2);
			row(input,0) = row(testInputs,t2);
			net(input);
			net.weightedParameterDerivative(input,coefficients,derivative);
			BOOST_REQUIRE_EQUAL(derivative.size(),numberOfParameters);
		}

		//estimate weighted derivative
		double epsilon=1.e-5;
		RealVector testDerivative(numberOfParameters,0.0);
		for(size_t w=0; w != numberOfParameters; ++w){
			//create points with an change of +-epsilon in the wth component
			RealVector point1(parameters);
			RealVector point2(parameters);
			point1(w)+=epsilon;
			point2(w)-=epsilon;
			//calculate first result
			net.setParameterVector(point1);
			//rerun the whole sequence
			net.resetInternalState();
			RealVector result1;
			for(size_t t2=0;t2 <=t; ++t2){
				result1=net(row(testInputs,t2));
			}
			//calculate second result
			net.setParameterVector(point2);
			//rerun the whole sequence
			net.resetInternalState();
			RealVector result2;
			for(size_t t2=0;t2 <=t; ++t2){
				result2=net(row(testInputs,t2));
			}

			//now estimate the derivative for the changed parameter
			testDerivative(w)+=inner_prod(row(coefficients,0),(result1-result2)/(2*epsilon));
		}
		std::cout<<"est: "<<testDerivative<<"\n calc:"<<derivative<<std::endl;
		//check wether the derivatives are identical
		BOOST_CHECK_SMALL(::shark::distance(derivative,testDerivative),epsilon);
	}

}