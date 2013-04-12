#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <cmath>

#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/Models/LinearModel.h>
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
//         RealMatrix estimatedHessian = estimateSecondDerivative(loss, testPoint, testLabel);

        //test eval
        double value = loss.eval(testLabel,testPoint);
        //std::cout<<value << " "<<valueResult<<std::endl;
        BOOST_CHECK_SMALL(value-valueResult, 1.e-13);

        //test evalDerivative (first)
        RealMatrix derivative;
        value = loss.evalDerivative(testLabel, testPoint, derivative);
        //std::cout<<derivative<< " " << estimatedDerivative<<std::endl;
        //std::cout<<testPoint << " "<<testLabel<<std::endl;
        BOOST_CHECK_SMALL(value - valueResult, 1.e-13);
        BOOST_CHECK_SMALL(norm_2(row(derivative,0) - estimatedDerivative), 1.e-5);

		//testEvalDerivative (second)
//		RealMatrix hessian;
//		value = loss.evalDerivative(testLabel, testPoint, derivative, hessian);
//		BOOST_CHECK_SMALL(value-valueResult, 1.e-13);
//		BOOST_CHECK_SMALL(norm_1(derivative - estimatedDerivative), 1.e-5);
//		double hessianDistance = norm_1(hessian - estimatedHessian);
//		BOOST_CHECK_SMALL(hessianDistance, 1.e-3);
        //std::cout<<derivative(0) / estimatedDerivative(0)<<" "<<hessian(0,0)<<" "<<estimatedHessian(0,0)<<std::endl;
    }
}
