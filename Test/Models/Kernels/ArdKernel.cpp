#define BOOST_TEST_MODULE Models_ArdKernel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "KernelDerivativeTestHelper.h"

#include <shark/Models/Kernels/ArdKernel.h>
#include <shark/Rng/GlobalRng.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Models_Kernels_ArdKernel)

BOOST_AUTO_TEST_CASE( DenseARDKernel_Parameters )
{
	const double gamma_init = 2.0;
	const unsigned int cur_dims = 3;
	DenseARDKernel kernel(cur_dims, gamma_init);

	// test gamma getter
	BOOST_CHECK_SMALL( gamma_init - kernel.gammaVector()(0), 1E-10);
	// test gamma setter
	RealVector new_gamma( cur_dims );
	new_gamma(0) = 9.0;
	new_gamma(1) = 16.0;
	new_gamma(2) = 25.0;
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
}

BOOST_AUTO_TEST_CASE( DenseARDKernel_Value )
{
	const unsigned int cur_dims = 3;
	double gamma_init = 2.0;
	DenseARDKernel kernel( cur_dims, gamma_init );
	// create testpoints
	RealVector x(cur_dims);
	x(0) = 0.4; x(1) = 1.1; x(2) = 0.55;
	RealVector z(cur_dims);
	z(0) = 0.22; z(1) = 1.0; z(2) = 0.8;

	//create test batches
	RealMatrix xBatch(10,cur_dims);
	RealMatrix zBatch(13,cur_dims);
	for(std::size_t i = 0; i != 10; ++i){
		xBatch(i,0) = Rng::uni(-3,3);
		xBatch(i,1) = Rng::uni(-3,3);
		xBatch(i,2) = Rng::uni(-3,3);
	}
	for(std::size_t i = 0; i != 13; ++i){
		zBatch(i,0) = Rng::uni(-3,3);
		zBatch(i,1) = Rng::uni(-3,3);
		zBatch(i,2) = Rng::uni(-3,3);
	}

	// check eval for default params
	double k_ret = kernel.eval(x, z);
	double my_ret = exp( - gamma_init * (  (x(0)-z(0))*(x(0)-z(0)) + (x(1)-z(1))*(x(1)-z(1)) + (x(2)-z(2))*(x(2)-z(2))  )  );
	BOOST_CHECK_SMALL( k_ret - my_ret, 1E-10);
	testEval(kernel,xBatch,zBatch);


	// set different params
	RealVector my_params(cur_dims);
	my_params(0) = 1.3;
	my_params(1) = 0.7;
	my_params(2) = 0.1;
	kernel.setParameterVector( my_params );

	// check eval for manual params
	k_ret = kernel.eval(x, z);
	my_ret = exp( -my_params(0)*my_params(0)*(x(0)-z(0))*(x(0)-z(0)) - my_params(1)*my_params(1)*(x(1)-z(1))*(x(1)-z(1)) - my_params(2)*my_params(2)*(x(2)-z(2))*(x(2)-z(2)) );
	BOOST_CHECK_SMALL( k_ret - my_ret, 1E-10);

	//check batches
	testEval(kernel,xBatch,zBatch);

	// everything again with a negative param as well:
	my_params(0) = -0.1;
	my_params(1) = 0.2;
	my_params(2) = -0.05;
	kernel.setParameterVector( my_params );

	// check eval for manual params
	k_ret = kernel.eval(x, z);
	my_ret = exp( -my_params(0)*my_params(0)*(x(0)-z(0))*(x(0)-z(0)) - my_params(1)*my_params(1)*(x(1)-z(1))*(x(1)-z(1)) - my_params(2)*my_params(2)*(x(2)-z(2))*(x(2)-z(2)) );
	BOOST_CHECK_SMALL( k_ret - my_ret, 1E-10);

	//check batches
	testEval(kernel,xBatch,zBatch);
}

BOOST_AUTO_TEST_CASE( DenseARDKernel_Derivative )
{
	const unsigned int cur_dims = 3;
	DenseARDKernel kernel(cur_dims, 4.0);

	// set different gammas
	RealVector my_gammas(cur_dims);
	my_gammas(0) = 1.3;
	my_gammas(1) = 0.7;
	my_gammas(2) = 0.1;
	kernel.setGammaVector( my_gammas );

	testKernelDerivative(kernel,3,1.e-5,1.e-4);
	testKernelInputDerivative(kernel,3,1.e-5,1.e-4);

	// set different params
	RealVector my_params(cur_dims);
	my_params(0) = 1.3;
	my_params(1) = 0.7;
	my_params(2) = 0.1;
	kernel.setParameterVector( my_params );

	testKernelDerivative(kernel,3,1.e-5,1.e-4);
	testKernelInputDerivative(kernel,3,1.e-5,1.e-4);

	// also set negative params
	my_params(0) = -0.1;
	my_params(1) = 0.2;
	my_params(2) = -0.05;
	kernel.setParameterVector( my_params );

	testKernelDerivative(kernel,3,1.e-5,1.e-4);
	testKernelInputDerivative(kernel,3,1.e-5,1.e-4);
}

BOOST_AUTO_TEST_SUITE_END()
