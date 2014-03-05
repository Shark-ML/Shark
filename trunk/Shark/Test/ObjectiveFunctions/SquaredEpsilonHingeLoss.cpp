#include <shark/ObjectiveFunctions/Loss/SquaredEpsilonHingeLoss.h>
#include <shark/Rng/GlobalRng.h>
#include "TestLoss.h"

#define BOOST_TEST_MODULE OBJECTIVEFUNCTIONS_SQUAREDEPSILONHINGELOSS
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;
using namespace std;

BOOST_AUTO_TEST_CASE( SQUAREDEPSILONHINGELOSS_EVAL ) {
	unsigned int maxTests = 10000;
	unsigned int minDim = 3;
	unsigned int maxDim = 10;
	for (unsigned int test = 0; test != maxTests; ++test) {
		double epsilon = Rng::uni(0,5);
		SquaredEpsilonHingeLoss loss(epsilon);

		std::size_t dim = Rng::discrete(minDim,maxDim);
		//sample point between -10,10
		RealMatrix testPoint(5,dim);
		RealMatrix testLabel(5,dim);
		RealVector valueResultV(5,0);
		for(std::size_t i = 0; i != 5; ++i){
			for(std::size_t j = 0; j != dim; ++j){
				testPoint(i,j) = Rng::uni(-10.0,10.0);
				testLabel(i,j) = Rng::uni(-10.0,10.0);
			}
			valueResultV(i) = 0.5*std::max(0.0, norm_sqr(row(testPoint,i)-row(testLabel,i))-sqr(epsilon));
		}
		double valueResult = sum(valueResultV);

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
			RealVector estimatedDerivative = estimateDerivative(loss, RealMatrix(rows(testPoint,i,i+1)), RealMatrix(rows(testLabel,i,i+1)));
			BOOST_CHECK_SMALL(norm_sqr(row(derivative,i) - estimatedDerivative), 1.e-5);
		}
	}
}
