#include <shark/ObjectiveFunctions/Loss/HingeLoss.h>
#include <shark/Rng/GlobalRng.h>
#include "TestLoss.h"

#define BOOST_TEST_MODULE OBJECTIVEFUNCTIONS_HINGELOSS
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;
using namespace std;

BOOST_AUTO_TEST_SUITE (ObjectiveFunctions_HingeLoss)

BOOST_AUTO_TEST_CASE( HINGELOSS_EVAL_TWOCLASS ) {
	unsigned int maxTests = 10000;
	for (unsigned int test = 0; test != maxTests; ++test) {
		HingeLoss loss;

		//sample point between -10,10
		RealMatrix testPoint(2,1);
		testPoint(0,0) = Rng::uni(-10.0,10.0);
		testPoint(1,0) = Rng::uni(-10.0,10.0);
		
		RealMatrix testPoint2D(2,2);
		testPoint2D(0,0) = testPoint(0,0);
		testPoint2D(0,1) = -testPoint(0,0);
		testPoint2D(1,0) = testPoint(1,0);
		testPoint2D(1,1) = -testPoint(1,0);
		testPoint2D*=-1;

		//sample label {-1,1}
		UIntVector testLabel(2);
		testLabel(0) = Rng::coinToss(0.5);
		testLabel(1) = Rng::coinToss(0.5);
		
		int label0 = testLabel(0)?1:-1;
		int label1 = testLabel(1)?1:-1;

		//the test results
		double valueResultP[] = {
			std::max(0.0, 1-testPoint(0,0)*label0),
			std::max(0.0, 1-testPoint(1,0)*label1)
		};
		double valueResult = valueResultP[0]+valueResultP[1];

		//test eval
		double value = loss.eval(testLabel,testPoint);
		double value2D = loss.eval(testLabel,testPoint2D);
		BOOST_CHECK_SMALL(value-valueResult, 1.e-13);
		BOOST_CHECK_SMALL(value2D-valueResult, 1.e-13);
		
		//test evalDerivative (first)
		RealMatrix derivative;
		RealMatrix derivative2D;
		value = loss.evalDerivative(testLabel, testPoint, derivative);
		value2D = loss.evalDerivative(testLabel, testPoint2D, derivative2D);
		BOOST_CHECK_SMALL(value - valueResult, 1.e-13);
		BOOST_CHECK_SMALL(value2D - valueResult, 1.e-13);
		BOOST_REQUIRE_EQUAL(derivative2D.size1(), 2);
		BOOST_REQUIRE_EQUAL(derivative2D.size2(),2);
		BOOST_REQUIRE_EQUAL(derivative.size1(), 2);
		BOOST_REQUIRE_EQUAL(derivative.size2(), 1);
		
		for(std::size_t i = 0; i != 2; ++i){
			RealVector estimatedDerivative = estimateDerivative(loss, RealMatrix(rows(testPoint,i,i+1)), subrange(testLabel,i,i+1));
			RealVector estimatedDerivative2D = estimateDerivative(loss, RealMatrix(rows(testPoint2D,i,i+1)), subrange(testLabel,i,i+1));
			if(std::abs(valueResultP[i])>1.e-5){
				BOOST_CHECK_SMALL(norm_2(row(derivative,i) - estimatedDerivative), 1.e-5);
				BOOST_CHECK_SMALL(norm_2(row(derivative2D,i) - estimatedDerivative2D), 1.e-5);
			}
			else{
				BOOST_CHECK_SMALL(norm_2(row(derivative,i)), 1.e-10);
				BOOST_CHECK_SMALL(norm_2(row(derivative2D,i)), 1.e-10);
			}
			
		}
	}
}

BOOST_AUTO_TEST_CASE( HINGELOSS_EVAL_MULTICLASS ) {
	unsigned int maxTests = 10000;
	unsigned int minDim = 3;
	unsigned int maxDim = 10;
	for (unsigned int test = 0; test != maxTests; ++test) {
		HingeLoss loss;

		std::size_t dim = Rng::discrete(minDim,maxDim);
		//sample point between -10,10
		RealMatrix testPoint(5,dim);
		UIntVector testLabel(5);
		RealVector valueResultP(5,0);
		for(std::size_t i = 0; i != 5; ++i){
			testLabel(i) = Rng::discrete(0,dim-1);
			testPoint(i,testLabel(i)) = Rng::uni(-10.0,10.0);
			for(std::size_t j = 0; j != dim; ++j){
				if(j == testLabel(i)) continue;
				testPoint(i,j) = Rng::uni(-10.0,10.0);
				valueResultP[i]+= std::max(0.0, 1-0.5*(testPoint(i,testLabel(i))- testPoint(i,j)));
			}
		}
		double valueResult = sum(valueResultP);

		//test eval
		double value = loss.eval(testLabel,testPoint);
		BOOST_CHECK_SMALL(value-valueResult, 1.e-13);
		
		//test evalDerivative (first)
		RealMatrix derivative;
		value = loss.evalDerivative(testLabel, testPoint, derivative);
		BOOST_CHECK_SMALL(value - valueResult, 1.e-13);
		BOOST_REQUIRE_EQUAL(derivative.size1(), 5);
		BOOST_REQUIRE_EQUAL(derivative.size2(), dim);
		
		for(std::size_t i = 0; i != 5; ++i){
			RealVector estimatedDerivative = estimateDerivative(loss, RealMatrix(rows(testPoint,i,i+1)), UIntVector(subrange(testLabel,i,i+1)));
			if(std::abs(valueResultP[i])>1.e-5){
				BOOST_CHECK_SMALL(norm_2(row(derivative,i) - estimatedDerivative), 1.e-5);
			}
			else if(valueResultP[i]<=0.0){
				BOOST_CHECK_SMALL(norm_2(row(derivative,i)), 1.e-10);
			}
			
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
