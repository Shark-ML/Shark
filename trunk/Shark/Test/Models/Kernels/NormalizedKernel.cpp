#define BOOST_TEST_MODULE Models_ArdKernel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "KernelDerivativeTestHelper.h"

#include <shark/Models/Kernels/NormalizedKernel.h>
#include <shark/Models/Kernels/PolynomialKernel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Rng/GlobalRng.h>

using namespace shark;

BOOST_AUTO_TEST_CASE( NormalizedKernel_Polynomial_Test )
{
	PolynomialKernel<> baseKernel(2,1);
	DenseNormalizedKernel kernel(&baseKernel);

	// test points
	RealVector x1(2);
	x1(0)=1;
	x1(1)=1;
	RealVector x2(2);
	x2(0)=-2;
	x2(1)=1;

	double result=baseKernel(x1,x2)/sqrt(baseKernel(x1,x1)*baseKernel(x2,x2));
	//evaluate point
	double test=kernel.eval(x1,x2);
	BOOST_CHECK_SMALL(result-test,1.e-15);

	testKernelInputDerivative(kernel, 2, 1e-7, 1e-2);
}

BOOST_AUTO_TEST_CASE( NormalizedKernel_GaussianRbf_Test )
{
	GaussianRbfKernel<> baseKernel(0.5);
	DenseNormalizedKernel kernel(&baseKernel);

	// test points
	RealVector x1(2);
	x1(0)=1;
	x1(1)=1;
	RealVector x2(2);
	x2(0)=-2;
	x2(1)=1;

	double result=baseKernel(x1,x2)/sqrt(baseKernel(x1,x1)*baseKernel(x2,x2));
	//evaluate point
	double test=kernel.eval(x1,x2);
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//test first derivative
	testKernelDerivative(kernel, 2, 1e-7, 1e-4);
	testKernelInputDerivative(kernel, 2, 1e-7, 1e-2);
}