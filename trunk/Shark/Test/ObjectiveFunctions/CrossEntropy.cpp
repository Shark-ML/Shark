#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>
#include <cmath>

#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/Models/LinearModel.h>
#include <shark/Rng/GlobalRng.h>
#include "TestLoss.h"

#define BOOST_TEST_MODULE OBJECTIVEFUNCTIONS_CROSSENTROPY
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>


using namespace shark;
using namespace std;

BOOST_AUTO_TEST_CASE( CROSSENTROPY_DERIVATIVES_TWO_CLASSES_SINGLE_INPUT ){
	unsigned int maxTests = 1000;
	for(unsigned int test = 0; test != maxTests; ++test){
		CrossEntropy loss;

		//sample point between -10,10
		RealMatrix testPoint(1,1);
		testPoint(0,0) = Rng::uni(-300.0,300.0);


		//sample label
		unsigned int label = Rng::coinToss();
		double calcLabel = label ? 1 : -1;
		UIntVector labelVec(1);
		labelVec(0) = label;
		//the test results
		double valueResult = std::log(1 + std::exp(- calcLabel * testPoint(0,0)));
		RealVector estimatedDerivative = estimateDerivative(loss, testPoint, labelVec);

		//test eval
		
		double value = loss.eval(labelVec,testPoint);
		BOOST_CHECK_SMALL(value-valueResult, 1.e-13);

		//test evalDerivative (first)
		RealMatrix derivative;
		value = loss.evalDerivative(labelVec, testPoint, derivative);
		BOOST_CHECK_SMALL(value - valueResult, 1.e-13);
		BOOST_CHECK_SMALL(norm_2(row(derivative,0) - estimatedDerivative), 1.e-8);
	}
}
BOOST_AUTO_TEST_CASE( CROSSENTROPY_DERIVATIVES_TWO_CLASSES_TWO_INPUT ){
	unsigned int maxTests = 10000;
	for(unsigned int test = 0; test != maxTests; ++test){
		CrossEntropy loss;

		//sample point between -10,10
		RealMatrix testPoint(1,2);
		testPoint(0,0) = Rng::uni(-150.0,150.0);
		testPoint(0,1) = -testPoint(0,0);


		//sample label
		unsigned int label = Rng::coinToss();
		UIntVector labelVec(1);
		labelVec(0) = label;
		//the test results
		double valueResult = std::log(1 + std::exp(-2* testPoint(0,label)));
		RealVector estimatedDerivative = estimateDerivative(loss, testPoint, labelVec);

		//test eval
		
		double value = loss.eval(labelVec,testPoint);
		BOOST_CHECK_SMALL(value-valueResult, 1.e-13);

		//test evalDerivative (first)
		RealMatrix derivative;
		value = loss.evalDerivative(labelVec, testPoint, derivative);
		BOOST_CHECK_SMALL(value - valueResult, 1.e-13);
		BOOST_CHECK_SMALL(norm_2(row(derivative,0) - estimatedDerivative), 1.e-8);
	}
}
BOOST_AUTO_TEST_CASE( CROSSENTROPY_DERIVATIVES_MULTI_CLASS ){
	unsigned int maxTests = 1000;
	for(unsigned int test = 0; test != maxTests; ++test){
		CrossEntropy loss;

		//sample point between -10,10
		
		RealMatrix testPoint(1,5);
		double norm = 0;
		for(std::size_t i = 0; i !=5; ++i){
			testPoint(0,i) = Rng::uni(-10.0,10.0);
			norm+=std::exp(testPoint(0,i));
		}


		//sample label
		unsigned int label = Rng::discrete(0,4);
		UIntVector labelVec(1);
		labelVec(0) = label;
		//std::cout<<testPoint(0,0)<<" "<<label<<" "<<calcLabel<<std::endl;
		//the test results
		double valueResult = std::log(norm)-testPoint(0,label);
		RealVector estimatedDerivative = estimateDerivative(loss, testPoint, labelVec);
// 		RealMatrix estimatedHessian = estimateSecondDerivative(loss, testPoint, label);

		//test eval
		
		double value = loss.eval(labelVec,testPoint);
		BOOST_CHECK_SMALL(value-valueResult, 1.e-13);

		//test evalDerivative (first)
		RealMatrix derivative;
		value = loss.evalDerivative(labelVec, testPoint, derivative);
		BOOST_CHECK_SMALL(value - valueResult, 1.e-13);
		BOOST_CHECK_SMALL(norm_2(row(derivative,0) - estimatedDerivative), 1.e-9);
		//std::cout<<derivative(0,0) / estimatedDerivative(0)<<std::endl;

		//testEvalDerivative (second)
// 		RealMatrix hessian;
// 		value = loss.evalDerivative(label, testPoint, derivative, hessian);
// 		BOOST_CHECK_SMALL(value-valueResult, 1.e-13);
// 		BOOST_CHECK_SMALL(norm_1(derivative - estimatedDerivative), 1.e-5);
// 		double hessianDistance = norm_1(hessian - estimatedHessian);
//  	BOOST_CHECK_SMALL(hessianDistance, 1.e-3);
//  	std::cout<<derivative(0) / estimatedDerivative(0)<<" "<<hessian(0,0)<<" "<<estimatedHessian(0,0)<<std::endl;
	}
}
