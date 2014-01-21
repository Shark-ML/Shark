#include <shark/Unsupervised/RBM/BinaryRBM.h>
#include <shark/Unsupervised/RBM/GradientApproximations/ExactGradient.h>
#include <shark/Unsupervised/RBM/analytics.h>

#define BOOST_TEST_MODULE RBM_ExactGradient
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "../ObjectiveFunctions/TestObjectiveFunction.h"

using namespace shark;
BOOST_AUTO_TEST_CASE( ExactGradient_NegLogLikelihood_MoreHidden )
{
	
	//create RBM with 8 visible and 16 hidden units
	BinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(8,16);
	
	
	//now test for several random subsets of possible training data and random initializations of the rbm
	for(std::size_t i = 0; i != 10; ++i){
		initRandomNormal(rbm,2);
		RealVector parameters=rbm.parameterVector();
		initRandomNormal(rbm,2);
		std::vector<RealVector> dataVec(50,RealVector(8));
		for(std::size_t j = 0; j != 50; ++j){
			for(std::size_t k = 0; k != 8; ++k){
				dataVec[j](k)=Rng::coinToss(0.5);
			}
		}
		UnlabeledData<RealVector> data = createDataFromRange(dataVec,25);
		
		//now calculate the test
		ExactGradient<BinaryRBM> gradient(&rbm);
		ExactGradient<BinaryRBM>::FirstOrderDerivative derivative;
		gradient.setData(data);
		double logProbTest = gradient.evalDerivative(parameters,derivative);
		long double logProb = negativeLogLikelihood(rbm,data)/50;
		BOOST_CHECK_CLOSE(logProbTest,logProb,1.e-5);
	}
}

BOOST_AUTO_TEST_CASE( ExactGradient_NegLogLikelihood_LessHidden )
{
	
	//create RBM with 8 visible and 16 hidden units
	BinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(8,4);
	
	
	//now test for several random subsets of possible training data and random initializations of the rbm
	for(std::size_t i = 0; i != 10; ++i){
		initRandomNormal(rbm,2);
		RealVector parameters=rbm.parameterVector();
		initRandomNormal(rbm,2);
		std::vector<RealVector> dataVec(50,RealVector(8));
		for(std::size_t j = 0; j != 50; ++j){
			for(std::size_t k = 0; k != 8; ++k){
				dataVec[j](k)=Rng::coinToss(0.5);
			}
		}
		UnlabeledData<RealVector> data = createDataFromRange(dataVec,25);
		
		//now calculate the test
		ExactGradient<BinaryRBM> gradient(&rbm);
		ExactGradient<BinaryRBM>::FirstOrderDerivative derivative;
		gradient.setData(data);
		double logProbTest = gradient.evalDerivative(parameters,derivative);
		long double logProb = negativeLogLikelihood(rbm,data)/50;
		BOOST_CHECK_CLOSE(logProbTest,logProb,1.e-5);
	}
}


BOOST_AUTO_TEST_CASE( ExactGradient_NegLogLikelihood_Gradient_MoreHidden )
{

	BinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(4,8);
	
	std::vector<RealVector> dataVec(50,RealVector(4));
	for(std::size_t j = 0; j != 50; ++j){
		for(std::size_t k = 0; k != 4; ++k){
			dataVec[j](k)=Rng::coinToss(0.5);
		}
	}
	UnlabeledData<RealVector> data = createDataFromRange(dataVec,25);
	
	for(std::size_t i = 0; i != 10; ++i){
		initRandomNormal(rbm,1);
		RealVector parameters=rbm.parameterVector();
		
		ExactGradient<BinaryRBM> gradient(&rbm);
		gradient.setData(data);
		testDerivative(gradient,parameters,1.e-2,1.e-10,0.1);
	}
}

BOOST_AUTO_TEST_CASE( ExactGradient_NegLogLikelihood_Gradient_LessHidden )
{

	BinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(8,4);
	
	std::vector<RealVector> dataVec(50,RealVector(8));
	for(std::size_t j = 0; j != 50; ++j){
		for(std::size_t k = 0; k != 8; ++k){
			dataVec[j](k)=Rng::coinToss(0.5);
		}
	}
	UnlabeledData<RealVector> data = createDataFromRange(dataVec,25);
	
	for(std::size_t i = 0; i != 10; ++i){
		initRandomNormal(rbm,2);
		RealVector parameters=rbm.parameterVector();
		
		ExactGradient<BinaryRBM> gradient(&rbm);
		gradient.setData(data);
		testDerivative(gradient,parameters,1.e-2,1.e-10,0.1);
	}
}
