#define BOOST_TEST_MODULE Kernels_ModelKernel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Models/Kernels/ModelKernel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/LinearModel.h>
#include "KernelDerivativeTestHelper.h"
#include <shark/Rng/GlobalRng.h>

using namespace shark;

BOOST_AUTO_TEST_CASE( ModelKernel_Parameters )
{
	double gamma = 0.2;
	GaussianRbfKernel<> kernel(0.2);
	LinearModel<> model(2,3);
	initRandomUniform(model,-1,1);
	RealVector modelParameters = model.parameterVector();
	
	std::size_t kernelParams = model.numberOfParameters()+1;
	
	//check whther the right parameters are returned
	{
		ModelKernel<RealVector> modelKernel(&kernel,&model);
		BOOST_REQUIRE_EQUAL(modelKernel.numberOfParameters(), kernelParams);
		RealVector kernelParameters = modelKernel.parameterVector();
		BOOST_REQUIRE_EQUAL(kernelParameters.size(), kernelParams);
		BOOST_CHECK_CLOSE(kernelParameters(0),gamma,1.e-14);
		for(std::size_t i = 0; i != modelParameters.size(); ++i){
			BOOST_CHECK_CLOSE(modelParameters(i),kernelParameters(i+1),1.e-14);
		}
	}
	//set parameters of kernel/model to 0 and check whether this also works
	{
		ModelKernel<RealVector> modelKernel(&kernel,&model);
		kernel.setGamma(0.1);
		model.setParameterVector(RealVector(model.numberOfParameters(),0.1));
		BOOST_REQUIRE_EQUAL(modelKernel.numberOfParameters(), kernelParams);
		RealVector kernelParameters = modelKernel.parameterVector();
		BOOST_REQUIRE_EQUAL(kernelParameters.size(), kernelParams);
		for(std::size_t i = 0; i != kernelParameters.size(); ++i){
			BOOST_CHECK_SMALL(kernelParameters(i)-0.1,1.e-14);
		}
	}
	//check whether setting parameters works
	{
		RealVector params(kernelParams);
		params(0) = gamma;
		subrange(params,1,kernelParams)=modelParameters;
		ModelKernel<RealVector> modelKernel(&kernel,&model);
		BOOST_REQUIRE_EQUAL(modelKernel.numberOfParameters(), kernelParams);
		modelKernel.setParameterVector(params);
		BOOST_CHECK_CLOSE(kernel.gamma(),gamma,1.e-14);
		RealVector modelParamsNew = model.parameterVector();
		for(std::size_t i = 0; i != modelParameters.size(); ++i){
			BOOST_CHECK_CLOSE(modelParameters(i),modelParamsNew(i),1.e-14);
		}
	}
}


BOOST_AUTO_TEST_CASE( ModelKernel_Eval_EvalDerivative )
{
	double gamma = 0.2;
	GaussianRbfKernel<> kernel(gamma);
	LinearModel<> model(2,3);
	initRandomUniform(model,-1,1);
	std::size_t kernelParams = model.numberOfParameters()+1;
	

	ModelKernel<RealVector> modelKernel(&kernel,&model);
	BOOST_REQUIRE_EQUAL(modelKernel.numberOfParameters(), kernelParams);

	//testpoints
	RealVector x1(2);
	x1(0)=1;
	x1(1)=1;
	RealVector x2(2);
	x2(0)=-2;
	x2(1)=0.0;
	//evaluate single point
	double test = modelKernel(x1,x2);
	double result = kernel(model(x1),model(x2));
	BOOST_CHECK_SMALL(result-test,1.e-15);
	
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
	testEval(modelKernel,batch1,batch2);

	//test first derivative
	testKernelDerivative(modelKernel,2,1.e-7,1.e-5);

}

