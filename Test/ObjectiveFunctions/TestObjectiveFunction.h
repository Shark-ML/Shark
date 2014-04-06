#ifndef SHARK_TEST_DERIVATIVETESTHELPER_H
#define SHARK_TEST_DERIVATIVETESTHELPER_H

#include <vector>
#include <shark/LinAlg/Base.h>
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include  <shark/Rng/GlobalRng.h>

namespace shark{
//estimates Derivative using the formula:
//df(x)/dx~=(f(x+e)-f(x-e))/2e
inline RealVector estimateDerivative(
	SingleObjectiveFunction& function,
	const RealVector& point,
	double epsilon=1.e-10
){
	RealVector gradient(point.size());
	for(std::size_t dim=0;dim!=point.size();++dim){
		RealVector testPoint1=point;
		testPoint1(dim)+=epsilon;
		double result1= function.eval(testPoint1);

		RealVector testPoint2=point;
		testPoint2(dim) -= epsilon;
		double result2 = function.eval(testPoint2);
		gradient[dim] = (result1-result2)/(2*epsilon);
	}
	return gradient;
}
inline RealMatrix estimateSecondDerivative(
	SingleObjectiveFunction& function,
	const RealVector& point,
	double epsilon=1.e-10
){
	typedef SingleObjectiveFunction::FirstOrderDerivative Derivative;
	
	RealMatrix hessian(point.size(),point.size());
	for(std::size_t dim=0;dim!=point.size();++dim){
		RealVector testPoint1 = point;
		testPoint1(dim) += epsilon;
		Derivative result1;
		function.evalDerivative(testPoint1,result1);

		RealVector testPoint2=point;
		testPoint2(dim) -= epsilon;
		Derivative result2;
		function.evalDerivative(testPoint2,result2);
		row(hessian,dim) = (result1-result2)/(2*epsilon);
	}
	return hessian;
}
void testDerivative(
	SingleObjectiveFunction& function, 
	const RealVector& point,
	double epsilon=1.e-10,
	double epsilonSecond = 1.e-10,
	double maxErrorPercentage = 0.001
){
	//double maxError = epsilon * 100;
	
	RealVector estimatedDerivative=estimateDerivative(function,point,epsilon);

	//calculate derivative and check that eval() and evalDerivative() match.
	typedef  SingleObjectiveFunction Function;
	Function::FirstOrderDerivative derivative;
	double resultEvalDerivative = function.evalDerivative(point,derivative);
	double resultEval = function.eval(point);
	BOOST_CHECK_CLOSE(resultEvalDerivative,resultEval, maxErrorPercentage);
	
	//calculate error between both
	BOOST_REQUIRE_EQUAL(estimatedDerivative.size(),derivative.size());
	//std::cout<<estimatedDerivative<<std::endl;
	//std::cout<<derivative<<std::endl;
	for(std::size_t i=0;i != estimatedDerivative.size(); ++i){
		if(derivative(i) > 0.01 && estimatedDerivative(i)>0.001)
			BOOST_CHECK_CLOSE(estimatedDerivative(i),derivative(i),maxErrorPercentage);
		//BOOST_CHECK_SMALL(estimatedDerivative(i) -derivative(i),maxError);
	}
	
	//if possible, calculate second derivative
	if(function.features() & Function::HAS_SECOND_DERIVATIVE){
		double maxErrorSecond = epsilonSecond * 100;
		
		RealMatrix estimatedHessian = estimateSecondDerivative(function,point,epsilonSecond);
		
		Function::SecondOrderDerivative secondDerivative;
		double resultEvalSecondDerivative = function.evalDerivative(point,secondDerivative);
		BOOST_CHECK_CLOSE(resultEvalSecondDerivative,resultEval, 1.e-5);
		//check first derivative again...
		BOOST_REQUIRE_EQUAL(estimatedDerivative.size(),secondDerivative.m_gradient.size());
		for(std::size_t i=0;i != estimatedDerivative.size(); ++i){
			BOOST_CHECK_CLOSE(estimatedDerivative(i),derivative(i),maxErrorPercentage);
			//BOOST_CHECK_SMALL(estimatedDerivative(i) - secondDerivative.m_gradient(i),maxError);
			//std::cout<<i<<" "<<estimatedDerivative(i)<<" "<<secondDerivative.m_gradient(i)<<std::endl;
		}
		
		//check second derivative
		BOOST_REQUIRE_EQUAL(estimatedHessian.size1(),secondDerivative.m_hessian.size1());
		BOOST_REQUIRE_EQUAL(estimatedHessian.size2(),secondDerivative.m_hessian.size2());
		for(std::size_t i=0;i != estimatedDerivative.size(); ++i){
			for(std::size_t j=0;j != estimatedDerivative.size(); ++j){
				BOOST_CHECK_SMALL(estimatedHessian(i,j) - secondDerivative.m_hessian(i,j),maxErrorSecond);
				//std::cout<<i<<" "<<j<<" "<<estimatedHessian(i,j)<<" "<<secondDerivative.m_hessian(i,j)<<std::endl;
			}
		}
	}
}


}
#endif
