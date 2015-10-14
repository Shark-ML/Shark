#define BOOST_TEST_MODULE RBM_ConvolutionalRBMBasic
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Unsupervised/RBM/ConvolutionalBinaryRBM.h>
using namespace shark;

BOOST_AUTO_TEST_SUITE (RBM_ConvolutionalRBMBasic)

BOOST_AUTO_TEST_CASE( Structure ){
	std::size_t inputSize1=6;
	std::size_t inputSize2=7;
	std::size_t visibleUnits = 42;
	
	std::size_t filterSize = 3;
	std::size_t numFilters = 5;
	std::size_t responseSize1 =4;
	std::size_t responseSize2 =5;
	std::size_t hiddenUnits = 4*5*numFilters;
	
	std::size_t numParams = hiddenUnits+visibleUnits + numFilters*filterSize*filterSize;
	
	
	//init RBMs and set structure
	ConvolutionalBinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(inputSize1,inputSize2,numFilters,filterSize);
	
	
	//check that setStructure actually created the right structure
	BOOST_CHECK_EQUAL(rbm.inputSize1(), inputSize1);
	BOOST_CHECK_EQUAL(rbm.inputSize2(), inputSize2);
	BOOST_CHECK_EQUAL(rbm.numberOfVN(), visibleUnits);
	BOOST_CHECK_EQUAL(rbm.visibleNeurons().size(), visibleUnits);
	
	BOOST_CHECK_EQUAL(rbm.filterSize1(), filterSize);
	BOOST_CHECK_EQUAL(rbm.filterSize2(), filterSize);
	BOOST_CHECK_EQUAL(rbm.numFilters(), numFilters);
	BOOST_CHECK_EQUAL(rbm.responseSize1(), responseSize1);
	BOOST_CHECK_EQUAL(rbm.responseSize2(), responseSize2);
	
	BOOST_CHECK_EQUAL(rbm.numberOfHN(), hiddenUnits);
	BOOST_CHECK_EQUAL(rbm.hiddenNeurons().size(), hiddenUnits);
	
	
	//test parameters
	BOOST_CHECK_EQUAL(rbm.numberOfParameters(), numParams);
	RealVector vector(numParams,0.0);
	for(std::size_t i = 0; i != numParams; ++i){
		vector(i)=i;
	}
	
	//check that if we set the right parameters, the correct one comes out again
	rbm.setParameterVector(vector);
	RealVector paramsRet=rbm.parameterVector();
	
	for(std::size_t i = 0; i != numParams; ++i){
		BOOST_CHECK_CLOSE(vector(i),paramsRet(i),1.e-14);
	}
	
	//now check, that all parameters are at their right places
	std::size_t currParam = 0;
	for(std::size_t i = 0; i != numFilters; ++i){
		for(std::size_t j = 0; j != filterSize; ++j){
			for(std::size_t k = 0; k != filterSize; ++k,++currParam){
				BOOST_CHECK_CLOSE(rbm.filters()[i](j,k), vector(currParam),1.e-14);
			}
		}
	}
	
	
}


BOOST_AUTO_TEST_CASE( Input ){
	std::size_t inputSize1=4;
	std::size_t inputSize2=4;
	std::size_t visibleUnits = 16;
	
	std::size_t filterSize = 3;
	std::size_t numFilters = 2;
	std::size_t hiddenUnits = 8;
	
	unsigned int filterIndizes[][9]={
		{0,1,2
		,4,5,6
		,8,9,10	
		},
		{1,2,3
		,5,6,7
		,9,10,11
		},
		{4,5,6
		,8,9,10
		,12,13,14
		},
		{5,6,7
		,9,10,11
		,13,14,15
		}
	};
	
	//init RBMs and set structure
	ConvolutionalBinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(inputSize1,inputSize2,numFilters,filterSize);
	initRandomNormal(rbm,1.0);
	
	RealMatrix W(hiddenUnits,visibleUnits,0.0);
	
	
	
	for(std::size_t i =0; i != 4; ++i){
		for(std::size_t j =0; j != 9; ++j){
			std::size_t x1 = j/3;
			std::size_t x2 = j%3;
			W(i,filterIndizes[i][j])=rbm.filters()[0](x1,x2);
			W(i+4,filterIndizes[i][j])=rbm.filters()[1](x1,x2);
		}
	}
	
	RealMatrix batchV(10,16);
	RealMatrix batchH(10,8);
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 16; ++j){
			batchV(i,j)=Rng::coinToss();
		}
		for(std::size_t j = 0; j != 8; ++j){
			batchH(i,j)=Rng::coinToss();
		}
	}
	
	RealMatrix resultWV=prod(batchV,trans(W));
	RealMatrix resultWH=prod(batchH,W);
	
	RealMatrix rbmWV(10,8,1.0);
	RealMatrix rbmWH(10,16,1.0);
	
	rbm.inputHidden(rbmWV,batchV);
	rbm.inputVisible(rbmWH,batchH);
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 16; ++j){
			BOOST_CHECK_CLOSE(rbmWH(i,j),resultWH(i,j),1.e-12);
		}
	}
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 8; ++j){
			BOOST_CHECK_CLOSE(rbmWV(i,j),resultWV(i,j),1.e-12);
		}
	}
}


BOOST_AUTO_TEST_SUITE_END()
