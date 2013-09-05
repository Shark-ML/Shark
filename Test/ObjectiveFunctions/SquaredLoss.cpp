#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Rng/GlobalRng.h>
#include "TestLoss.h"

#define BOOST_TEST_MODULE OBJECTIVEFUNCTIONS_SQUAREDLOSS
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;
using namespace std;

BOOST_AUTO_TEST_CASE( SQUAREDLOSS_EVAL ) {
	unsigned int maxTests = 10000;
	for (unsigned int test = 0; test != maxTests; ++test) {
		SquaredLoss<> loss;

		//sample point between -10,10
		RealMatrix testPoint(1,2);
		testPoint(0,0) = Rng::uni(-10.0,10.0);
		testPoint(0,1) = Rng::uni(-10.0,10.0);

		//sample label between -10,10
		RealMatrix testLabel(1,2);
		testLabel(0,0) = Rng::uni(-10.0,10.0);
		testLabel(0,1) = Rng::uni(-10.0,10.0);


		//the test results
		double valueResult = sqr(testPoint(0,0)-testLabel(0,0))+sqr(testPoint(0,1)-testLabel(0,1));
		RealVector estimatedDerivative = estimateDerivative(loss, testPoint, testLabel);

		//test eval
		double value = loss.eval(testLabel,testPoint);
		BOOST_CHECK_SMALL(value-valueResult, 1.e-13);

		//test evalDerivative (first)
		RealMatrix derivative;
		value = loss.evalDerivative(testLabel, testPoint, derivative);
		BOOST_CHECK_SMALL(value - valueResult, 1.e-13);
		BOOST_CHECK_SMALL(norm_2(row(derivative,0) - estimatedDerivative), 1.e-5);
	}
}

BOOST_AUTO_TEST_CASE( SQUAREDLOSS_EVAL_Classification ) {
	unsigned int maxTests = 10000;
	for (unsigned int test = 0; test != maxTests; ++test) {
		SquaredLoss<RealVector,unsigned int> loss;
		SquaredLoss<RealVector,RealVector> lossOneHot;

		//sample point between -10,10
		RealMatrix testPoint(1,3);
		testPoint(0,0) = Rng::uni(-10.0,10.0);
		testPoint(0,1) = Rng::uni(-10.0,10.0);
		testPoint(0,2) = Rng::uni(-10.0,10.0);

		//sample class label
		UIntVector testLabelDisc(1);
		testLabelDisc(0) = Rng::discrete(0,2);
		
		RealMatrix testLabel(1,3);
		testLabel(0,testLabelDisc(0))=1;


		//the test results
		double valueResult = lossOneHot.eval(testLabel,testPoint);
		RealVector estimatedDerivative = estimateDerivative(lossOneHot, testPoint, testLabel);

		//test eval
		double value = loss.eval(testLabelDisc,testPoint);
		BOOST_CHECK_SMALL(value-valueResult, 1.e-13);

		//test evalDerivative (first)
		RealMatrix derivative;
		value = loss.evalDerivative(testLabelDisc, testPoint, derivative);
		BOOST_CHECK_SMALL(value - valueResult, 1.e-13);
		BOOST_CHECK_SMALL(norm_2(row(derivative,0) - estimatedDerivative), 1.e-5);
	}
}
