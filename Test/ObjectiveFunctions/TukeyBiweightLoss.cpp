#include <shark/ObjectiveFunctions/Loss/TukeyBiweightLoss.h>
#include <shark/Rng/GlobalRng.h>
#include "TestLoss.h"

#define BOOST_TEST_MODULE OBJECTIVEFUNCTIONS_TUKEYBIWEIGHTLOSS
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;
using namespace std;

BOOST_AUTO_TEST_CASE( HUBERLOSS_TEST ) {
	unsigned int maxTests = 10000;
	for (unsigned int test = 0; test != maxTests; ++test) {
		TukeyBiweightLoss loss(1.5);

		RealMatrix testPoint(2,3);
		testPoint(0,0) = Rng::uni(-1,1);
		testPoint(1,0) = Rng::uni(-1,1);
		testPoint(0,1) = Rng::uni(-1,1);
		testPoint(1,1) = Rng::uni(-1,1);
		testPoint(0,2) = Rng::uni(-1,1);
		testPoint(1,2) = Rng::uni(-1,1);
		
		RealMatrix testLabel(2,3);
		testLabel(0,0) = Rng::uni(-1,1);
		testLabel(1,0) = Rng::uni(-1,1);
		testLabel(0,1) = Rng::uni(-1,1);
		testLabel(1,1) = Rng::uni(-1,1);
		testLabel(0,2) = Rng::uni(-1,1);
		testLabel(1,2) = Rng::uni(-1,1);
		
		//test evalDerivative (first)
		RealMatrix derivative;
		double valueEval = loss.eval(testLabel, testPoint);
		double valueEvalDerivative = loss.evalDerivative(testLabel, testPoint, derivative);
		BOOST_CHECK_SMALL(valueEval - valueEvalDerivative, 1.e-13);
		BOOST_REQUIRE_EQUAL(derivative.size1(), 2);
		BOOST_REQUIRE_EQUAL(derivative.size2(), 3);
		
		for(std::size_t i = 0; i != 2; ++i){
			double value = loss(row(testLabel,i), row(testPoint,i));
			BOOST_CHECK_SMALL(value, sqr(1.5)/6.0+1.e-5);
		}
		
		for(std::size_t i = 0; i != 2; ++i){
			RealVector estimatedDerivative = estimateDerivative(loss, RealMatrix(rows(testPoint,i,i+1)), rows(testLabel,i,i+1));
			BOOST_CHECK_SMALL(norm_2(row(derivative,i) - estimatedDerivative), 1.e-5);
		}
	}
}