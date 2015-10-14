#ifndef TEST_OBJECTIVEFUNCTIONS_TESTLOSS
#define TEST_OBJECTIVEFUNCTIONS_TESTLOSS

namespace shark{

template<class Loss, class Point,class Label>
RealVector estimateDerivative(
	Loss const& loss,
	Point const& point,
	Label const& label,
	double epsilon = 1.e-5
){
	RealVector gradient(point.size2());
	for(size_t parameter = 0; parameter != point.size2(); ++parameter){
		Point testPoint1 = point;
		testPoint1(0,parameter) += epsilon;
		double result1 = loss.eval(label, testPoint1);

		Point testPoint2 = point;
		testPoint2(0,parameter) -= epsilon;
		double result2 = loss.eval(label, testPoint2);

		gradient(parameter)=(result1 - result2) / (2 * epsilon);
	}
	return gradient;
}
template<class Loss,class Point,class Label>
RealMatrix estimateSecondDerivative(
	Loss const& loss,
	Point const& point,
	Label const& label,
	double epsilon = 1.e-5
){
	RealMatrix hessian(point.size2(), point.size2());
	hessian.clear();
	for(size_t parameter = 0;parameter != point.size2(); ++parameter){
		Point testPoint1 = point;
		testPoint1(0,parameter) += epsilon;
		RealVector grad1 = estimateDerivative(loss, testPoint1, label);

		Point testPoint2 = point;
		testPoint2(0,parameter) -= epsilon;
		RealVector grad2 = estimateDerivative(loss, testPoint2, label);

		row(hessian,parameter)=(grad1 - grad2) / (2 * epsilon);
	}
	return hessian;
}
}

#endif
