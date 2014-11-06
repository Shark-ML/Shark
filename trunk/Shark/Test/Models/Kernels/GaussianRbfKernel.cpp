#define BOOST_TEST_MODULE Kernels_GaussianRbfKernel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "KernelDerivativeTestHelper.h"

#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Core/Timer.h>
using namespace shark;

template<class Matrix,class Kernel>
void testBatch(Kernel& kernel,std::size_t batchSize1,std::size_t batchSize2,std::size_t dim){
	//generate two sets of points and check that the batched results equal the single results
	Matrix batch1(batchSize1,dim);
	Matrix batch2(batchSize2,dim);
	for(std::size_t i = 0; i != batchSize1;++i){
		for(std::size_t j = 0; j != dim; ++j)
			batch1(i,j)=Rng::uni(-1,1);
	}
	for(std::size_t i = 0; i != batchSize2;++i){
		for(std::size_t j = 0; j != dim; ++j)
			batch2(i,j)=Rng::uni(-1,1);
	}
	testEval(kernel,batch1,batch2);

}

BOOST_AUTO_TEST_SUITE (Models_Kernels_GaussianRbfKernel)

BOOST_AUTO_TEST_CASE( DenseRbfKernel_Test )
{
	const double gamma1=0.1;
	const double gamma2=0.2;
	DenseRbfKernel kernel(gamma1);

	//firs evaluate a single test point
	//testpoints
	RealVector x1(2);
	x1(0)=1;
	x1(1)=0;
	RealVector x2(2);
	x2(0)=2;
	x2(1)=0;

	double result=std::exp(-gamma1);
	//evaluate point
	double test=kernel.eval(x1,x2);
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//now the same with parameterVector to ensure that gamma is set correctly
	RealVector testParameter(1);
	testParameter(0)=gamma2;
	kernel.setParameterVector(testParameter);
	BOOST_CHECK_SMALL(gamma2-kernel.gamma(),1.e-15);

	//now test whether the parametervector is created correctly
	RealVector parameter=kernel.parameterVector();
	BOOST_CHECK_SMALL(norm_sqr(parameter-testParameter),1.e-15);

	//evaluate point for the new gamma value
	test=kernel.eval(x1,x2);
	result=std::exp(-gamma2);
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//test batches
	kernel.setGamma(gamma1);
	testBatch<RealMatrix>(kernel,100,200,2);
	testBatch<RealMatrix>(kernel,100,1,2);

	//test derivatives
	testKernelDerivative(kernel,2,1.e-7);
	testKernelInputDerivative(kernel,2);
}
BOOST_AUTO_TEST_CASE( DenseRbfKernelUnconstrained_Test )
{
	const double gamma1=0.1;
	const double gamma2=0.2;
	DenseRbfKernel kernel( gamma1, true );

	//firs evaluate a single test point
	//testpoints
	RealVector x1(2);
	x1(0)=1;
	x1(1)=0;
	RealVector x2(2);
	x2(0)=2;
	x2(1)=0;

	double result=std::exp(-gamma1);
	//evaluate point
	double test=kernel.eval(x1,x2);
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//now the same with parameterVector to ensure that gamma is set correctly
	RealVector testParameter(1);
	testParameter(0)=gamma2;
	kernel.setParameterVector(testParameter);
	BOOST_CHECK_SMALL( gamma2 - std::log(kernel.gamma()) , 1.e-15 );

	//now test whether the parametervector is created correctly
	RealVector parameter=kernel.parameterVector();
	BOOST_CHECK_SMALL(norm_sqr(parameter-testParameter),1.e-15);

	//evaluate point for the new gamma value
	test=kernel.eval(x1,x2);
	result=std::exp( -std::exp(gamma2) );
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//test batches
	kernel.setGamma(gamma1);
	testBatch<RealMatrix>(kernel,100,200,2);
	testBatch<RealMatrix>(kernel,100,1,2);

	//test derivatives
	testKernelDerivative(kernel,2,1.e-7);
	testKernelInputDerivative(kernel,2);
}

BOOST_AUTO_TEST_CASE( CompressedRbfKernel_Test )
{
	const double gamma=0.1;
	CompressedRbfKernel kernel(gamma);

	//testpoints
	CompressedRealVector x1(17000);
	x1(3745)=1;
	x1(14885)=0;
	CompressedRealVector x2(17000);
	x2(3745)=1;
	x2(14885)=-1;

	double result=std::exp(-gamma);
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
	BOOST_CHECK_SMALL(norm_sqr(parameter-testParameter),1.e-15);

	//test first derivative
	testKernelDerivative(kernel,2);
	//doesn't make sense for sparse (also does not work)
//	testKernelInputDerivative(kernel,2);

}
BOOST_AUTO_TEST_CASE( CompressedRbfKernelUnconstrained_Test )
{
	const double gamma=0.1;
	CompressedRbfKernel kernel( gamma, true );

	//testpoints
	CompressedRealVector x1(17000);
	x1(3745)=1;
	x1(14885)=0;
	CompressedRealVector x2(17000);
	x2(3745)=1;
	x2(14885)=-1;

	double result=std::exp(-gamma);
	//evaluate point
	double test=kernel.eval(x1,x2);
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//now the same with parameterVector to ensure that gmame is set correctly
	RealVector testParameter(1);
	testParameter(0)=gamma;
	kernel.setParameterVector(testParameter);

	//evaluate point
	test=kernel.eval(x1,x2);
	result=std::exp( -std::exp(gamma) );
	BOOST_CHECK_SMALL(result-test,1.e-15);

	//now test whether the parametervector is created correctly
	RealVector parameter=kernel.parameterVector();
	BOOST_CHECK_SMALL(norm_sqr(parameter-testParameter),1.e-15);

	//test first derivative
	testKernelDerivative(kernel,2);
	//doesn't make sense for sparse (also does not work)
//	testKernelInputDerivative(kernel,2);

}

#ifdef NDEBUG

void benchmark(std::size_t batchSize1,std::size_t batchSize2, std::size_t dim, std::size_t iterations){
	const double gamma1=0.1;
	DenseRbfKernel kernel(gamma1);

	RealMatrix batch1(batchSize1,dim);
	RealMatrix batch2(batchSize2,dim);
	std::vector<RealVector> batchVec1(batchSize1,RealVector(dim));
	std::vector<RealVector> batchVec2(batchSize2,RealVector(dim));
	for(std::size_t i = 0; i != batchSize1;++i){
		for(std::size_t j = 0; j != dim; ++j)
			batchVec1[i](j)=batch1(i,j)=FastRng::uni(-1,1);
	}
	for(std::size_t i = 0; i != batchSize2;++i){
		for(std::size_t j = 0; j != dim; ++j)
			batchVec2[i](j)=batch2(i,j)=Rng::uni(-1,1);
	}


	RealMatrix result(batchSize1,batchSize2);
	result.clear();
	double start1=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		kernel.eval(batch1,batch2,result);
	}
	double end1=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		for(std::size_t i = 0; i != batchSize1; ++i){
			for(std::size_t j = 0; j != batchSize2; ++j){
				result(i,j)+=kernel.eval(batchVec1[i],batchVec2[j]);
			}
		}
	}
	double end2=Timer::now();
	double speedupFactor = (end2-end1)/(end1-start1);
	std::cout<<batchSize1<<"\t"<<end1-start1<<"\t  "<<end2-end1<<"\t  "<<speedupFactor<<" \t  "<<sum(result)<<std::endl;
}
BOOST_AUTO_TEST_CASE( DenseRbfKernel_BENCHMARK )
{
	std::cout<<"Benchmarking"<<std::endl;
	std::size_t batchSize1=50;
	std::size_t batchSize2=50;
	std::size_t batchSize2Inc=10;
	std::size_t batchSize1Inc=10;
	std::size_t dim=400;
	std::size_t iterations=1;
	std::size_t steps=50;
	std::cout<<"starting benchmark 2batches"<<std::endl;
	std::cout<<"batchsize     timeBatch    timeSingle Speedup"<<std::endl;
	for(std::size_t step = 0; step != steps; ++step){
		benchmark(batchSize1+step*batchSize1Inc,batchSize2+step*batchSize2Inc,dim,iterations);
	}
}
#endif

BOOST_AUTO_TEST_SUITE_END()
