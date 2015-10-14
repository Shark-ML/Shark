#define BOOST_TEST_MODULE Kernels_PolynomialKernel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Models/Kernels/PolynomialKernel.h>
#include "KernelDerivativeTestHelper.h"
#include <shark/Rng/GlobalRng.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Models_Kernels_PolynomialKernel)

BOOST_AUTO_TEST_CASE( DensePolynomialKernel_Value ){
	double result3 = 64.0;
	double result2 = 16.0;
	double result1 = 4.0;
	DensePolynomialKernel kernel(3, 1.0, false);

	// test points
	RealVector x1(2);
	x1(0) = 1.5;
	x1(1) = 0.0;
	RealVector x2(2);
	x2(0) = 2.0;
	x2(1) = 1.0;
	//evaluate single point
	double test = kernel(x1,x2);
	BOOST_CHECK_SMALL(result3-test,1.e-15);
	
	//evaluate single point
	kernel.setDegree(2);
	test = kernel(x1,x2);
	BOOST_CHECK_SMALL(result2-test,1.e-15);
	
	kernel.setDegree(1);
	test = kernel(x1,x2);
	BOOST_CHECK_SMALL(result1-test,1.e-15);
	
	//evaluate a batch of points
	RealMatrix batch1(10,2);
	RealMatrix batch2(20,2);
	for(std::size_t i = 0; i != 10;++i){
		for(std::size_t j = 0; j != 2; ++j)
			batch1(i,j)=Rng::uni(-1,1);
	}
	for(std::size_t i = 0; i != 20;++i){
		for(std::size_t j = 0; j != 2; ++j)
			batch2(i,j)=Rng::uni(-1,1);
	}
	kernel.setDegree(3);
	testEval(kernel,batch1,batch2);
	kernel.setDegree(2);
	testEval(kernel,batch1,batch2);
	kernel.setDegree(1);
	testEval(kernel,batch1,batch2);
	
	//test first derivative
	kernel.setDegree(1);
	testKernelDerivative(kernel,2,1.e-7,1.e-5);
	testKernelInputDerivative(kernel,2,1.e-7,1.e-5);
	
	kernel.setDegree(2);
	testKernelDerivative(kernel,2,1.e-7,1.e-5);
	testKernelInputDerivative(kernel,2,1.e-7,1.e-5);
}

BOOST_AUTO_TEST_CASE( SparsePolynomialKernel_Test ){
	double result3 = 64.0;
	double result2 = 16.0;
	double result1 = 4.0;
	CompressedPolynomialKernel kernel(3, 1.0, false);

	// test points
	CompressedRealVector x1(200);
	x1(30) = 1.5;
	CompressedRealVector x2(200);
	x2(30) = 2.0;
	x2(51) = 1.0;
	//evaluate single point
	double test = kernel(x1,x2);
	BOOST_CHECK_SMALL(result3-test,1.e-15);
	
	//evaluate single point
	kernel.setDegree(2);
	test = kernel(x1,x2);
	BOOST_CHECK_SMALL(result2-test,1.e-15);
	
	kernel.setDegree(1);
	test = kernel(x1,x2);
	BOOST_CHECK_SMALL(result1-test,1.e-15);
	
	//evaluate a batch of points
	CompressedRealMatrix batch1(10,200);
	CompressedRealMatrix batch2(20,200);
	for(std::size_t i = 0; i != 10;++i){
		for(std::size_t j = 0; j != 2; ++j)
			batch1(i,j*100)=Rng::uni(-1,1);
	}
	for(std::size_t i = 0; i != 20;++i){
		for(std::size_t j = 0; j != 4; ++j)
			batch2(i,j*50)=Rng::uni(-1,1);
	}
	kernel.setDegree(3);
	testEval(kernel,batch1,batch2);
	kernel.setDegree(2);
	testEval(kernel,batch1,batch2);
	kernel.setDegree(1);
	testEval(kernel,batch1,batch2);

	//test first derivative, input derivative for sparse inputs is not-so-usefull
	kernel.setDegree(1);
	testKernelDerivative(kernel,2,1.e-7,1.e-5);
	
	kernel.setDegree(2);
	testKernelDerivative(kernel,2,1.e-7,1.e-5);

}

BOOST_AUTO_TEST_SUITE_END()
