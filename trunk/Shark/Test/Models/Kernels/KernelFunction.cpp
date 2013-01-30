
#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#define BOOST_TEST_MODULE ML_KernelFunction
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "KernelDerivativeTestHelper.h"

#include <boost/math/constants/constants.hpp>

#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/MonomialKernel.h>
#include <shark/Models/Kernels/PolynomialKernel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/ArdKernel.h>
#include <shark/Models/Kernels/NormalizedKernel.h>
#include <shark/Models/Kernels/ScaledKernel.h>
#include <shark/Models/Kernels/MklKernel.h>
#include  <shark/Rng/GlobalRng.h>
#include <cmath>

using namespace shark;




BOOST_AUTO_TEST_CASE( DenseRbfKernel_Test )
{
	const double gamma=0.1;
	DenseRbfKernel kernel(gamma);


	BOOST_CHECK_SMALL( gamma - kernel.gamma(), 1.e-15 );
	kernel.setGamma( 0.5*gamma );
	BOOST_CHECK_SMALL( 0.5*gamma - kernel.gamma(), 1.e-15 );
	kernel.setGamma( gamma );

	//testpoints
	RealVector x1(2);
	x1(0)=1;
	x1(1)=0;
	RealVector x2(2);
	x2(0)=2;
	x2(1)=0;

	double result=exp(-gamma);
	//evaluate point
	double test=kernel.eval(x1,x2);
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//now the same with parameterVector to ensure that gamma is set correctly
	RealVector testParameter(1);
	testParameter(0)=gamma;
	kernel.setParameterVector(testParameter);

	//evaluate point
	test=kernel.eval(x1,x2);
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//now test whether the parametervector is created correctly
	RealVector parameter=kernel.parameterVector();
	BOOST_CHECK_SMALL(normSqr(parameter-testParameter),1.e-15);

	//test derivatives
	testKernelDerivative(kernel,2);
	testKernelInputDerivative(kernel,2);
}
BOOST_AUTO_TEST_CASE( DenseRbfKernel_Unconstrained_Test )
{
	const double gamma=0.1;
	DenseRbfKernel kernel(gamma, true);


	BOOST_CHECK_SMALL( gamma - kernel.gamma(), 1.e-15 );
	kernel.setGamma( 0.5*gamma );
	BOOST_CHECK_SMALL( 0.5*gamma - kernel.gamma(), 1.e-15 );
	kernel.setGamma( gamma );

	//testpoints
	RealVector x1(2);
	x1(0)=1;
	x1(1)=0;
	RealVector x2(2);
	x2(0)=2;
	x2(1)=0;

	double result=exp(-gamma);
	//evaluate point
	double test=kernel.eval(x1,x2);
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//now the same with parameterVector to ensure that gamma is set correctly
	RealVector testParameter(1);
	testParameter(0) = gamma;
	kernel.setParameterVector(testParameter);

	//evaluate point
	test=kernel.eval(x1,x2);
	result=exp( -std::exp(gamma) );
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//now test whether the parametervector is created correctly
	RealVector parameter=kernel.parameterVector();
	BOOST_CHECK_SMALL(normSqr(parameter-testParameter),1.e-15);

	//test derivatives
	testKernelDerivative(kernel,2);
	testKernelInputDerivative(kernel,2);
}

BOOST_AUTO_TEST_CASE( CompressedRbfKernel_Test )
{
	const double gamma=0.1;
	CompressedRbfKernel kernel(gamma);

	BOOST_CHECK_SMALL( gamma - kernel.gamma(), 1.e-15 );
	kernel.setGamma( 0.5*gamma );
	BOOST_CHECK_SMALL( 0.5*gamma - kernel.gamma(), 1.e-15 );
	kernel.setGamma( gamma );

	//testpoints
	CompressedRealVector x1(17000);
	x1(3745)=1;
	x1(14885)=0;
	CompressedRealVector x2(17000);
	x2(3745)=1;
	x2(14885)=-1;

	double result=exp(-gamma);
	//evaluate point
	double test=kernel.eval(x1,x2);
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//now the same with parameterVector to ensure that gmame is set correctly
	RealVector testParameter(1);
	testParameter(0)=gamma;
	kernel.setParameterVector(testParameter);

	//evaluate point
	test=kernel.eval(x1,x2);
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//now test whether the parametervector is created correctly
	RealVector parameter=kernel.parameterVector();
	BOOST_CHECK_SMALL(normSqr(parameter-testParameter),1.e-15);

	//test first derivative
	testKernelDerivative(kernel,2);
	testKernelInputDerivative(kernel,2);

}
BOOST_AUTO_TEST_CASE( CompressedRbfKernel_Unconstrained_Test )
{
	const double gamma=0.1;
	CompressedRbfKernel kernel(gamma, true);

	BOOST_CHECK_SMALL( gamma - kernel.gamma(), 1.e-15 );
	kernel.setGamma( 0.5*gamma );
	BOOST_CHECK_SMALL( 0.5*gamma - kernel.gamma(), 1.e-15 );
	kernel.setGamma( gamma );

	//testpoints
	CompressedRealVector x1(17000);
	x1(3745)=1;
	x1(14885)=0;
	CompressedRealVector x2(17000);
	x2(3745)=1;
	x2(14885)=-1;

	double result=exp(-gamma);
	//evaluate point
	double test=kernel.eval(x1,x2);
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//now the same with parameterVector to ensure that gmame is set correctly
	RealVector testParameter(1);
	testParameter(0)=gamma;
	kernel.setParameterVector(testParameter);

	//evaluate point
	result=exp( -std::exp(gamma) );
	test=kernel.eval(x1,x2);
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//now test whether the parametervector is created correctly
	RealVector parameter=kernel.parameterVector();
	BOOST_CHECK_SMALL(normSqr(parameter-testParameter),1.e-15);

	//test first derivative
	testKernelDerivative(kernel,2);
	testKernelInputDerivative(kernel,2);
}

BOOST_AUTO_TEST_CASE( DenseARDKernel_Test )
{
	const unsigned int cur_dims = 3;
	const double gamma_init = 4.0;
	DenseARDKernel kernel( cur_dims, gamma_init );
	// create testpoints
	RealVector x(cur_dims);
	x(0) = 0.4; x(1) = 1.1; x(2) = 0.55;
	RealVector z(cur_dims);
	z(0) = 0.22; z(1) = 1.0; z(2) = 0.8;
	// check eval for default gammas
	double k_ret = kernel.eval(x, z);
	double my_ret = exp( -  gamma_init * (  (x(0)-z(0))*(x(0)-z(0)) + (x(1)-z(1))*(x(1)-z(1)) + (x(2)-z(2))*(x(2)-z(2))  )  );
	BOOST_CHECK_SMALL( k_ret - my_ret, 1E-10);
	// test gamma getter
	BOOST_CHECK_SMALL( gamma_init - kernel.gammaVector()(0), 1E-10);
	// test gamma setter
	RealVector new_gamma(3); new_gamma(0) = 9.0; new_gamma(1) = 16.0; new_gamma(2) = 25.0;
	kernel.setGammaVector( new_gamma );
	RealVector check_gamma = kernel.gammaVector();
	for ( std::size_t i=0; i<new_gamma.size(); i++ ) {
		BOOST_CHECK_SMALL( new_gamma(i)-check_gamma(i), 1E-10);
	}
	RealVector check_p = kernel.parameterVector();
	for ( std::size_t i=0; i<check_p.size(); i++ ) {
		BOOST_CHECK_SMALL( check_p(i)-std::sqrt(check_gamma(i)), 1E-10);
	}


	// set different parameters
	RealVector my_params(cur_dims);
	my_params(0) = 1.3;
	my_params(1) = 0.7;
	my_params(2) = 0.1;
	kernel.setParameterVector( my_params );
	// test if parameters are set correctly
	BOOST_CHECK_EQUAL( cur_dims, kernel.numberOfParameters() );
	RealVector k_params;
	k_params = kernel.parameterVector();
	for ( unsigned int i=0; i<my_params.size(); i++ )
		BOOST_CHECK_SMALL( my_params(i) - k_params(i), 1e-13 );
	// test if gamma is set correctly
	RealVector new_gammas = kernel.gammaVector();
	for ( unsigned int i=0; i<my_params.size(); i++ )
		BOOST_CHECK_SMALL( my_params(i)*my_params(i) - new_gammas(i), 1e-13 );
	// check eval for manual params
	k_ret = kernel.eval(x, z);
	my_ret = exp( -my_params(0)*my_params(0)*(x(0)-z(0))*(x(0)-z(0)) - my_params(1)*my_params(1)*(x(1)-z(1))*(x(1)-z(1)) - my_params(2)*my_params(2)*(x(2)-z(2))*(x(2)-z(2)) );
	BOOST_CHECK_SMALL( k_ret - my_ret, 1E-10);
	// check first derivative
	testKernelDerivative(kernel,3,1.e-5,1.e-4);
	testKernelInputDerivative(kernel,3,1.e-5,1.e-4);

	// everything again with a negative param as well:
	my_params(0) = -0.1;
	my_params(1) = 0.2;
	my_params(2) = -0.05;
	kernel.setParameterVector( my_params );
	// test if parameters are set correctly
	k_params = kernel.parameterVector();
	for ( unsigned int i=0; i<my_params.size(); i++ )
		BOOST_CHECK_SMALL( my_params(i) - k_params(i), 1e-13 );
	// test if gamma is set correctly
	new_gammas = kernel.gammaVector();
	for ( unsigned int i=0; i<my_params.size(); i++ )
		BOOST_CHECK_SMALL( my_params(i)*my_params(i) - new_gammas(i), 1e-13 );
	// check eval for manual gammas
	k_ret = kernel.eval(x, z);
	my_ret = exp( -my_params(0)*my_params(0)*(x(0)-z(0))*(x(0)-z(0)) - my_params(1)*my_params(1)*(x(1)-z(1))*(x(1)-z(1)) - my_params(2)*my_params(2)*(x(2)-z(2))*(x(2)-z(2)) );
	BOOST_CHECK_SMALL( k_ret - my_ret, 1E-10);
	// check first derivative
	testKernelDerivative(kernel,3,1.e-5,1.e-4);
	testKernelInputDerivative(kernel,3,1.e-5,1.e-4);
}

BOOST_AUTO_TEST_CASE( DenseNormalizedKernel_Test )
{
	const double gamma = 0.1;
	DenseRbfKernel baseKernel(gamma);
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