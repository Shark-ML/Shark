#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/SubrangeKernel.h>
#include <shark/Models/Kernels/KernelHelpers.h>

#define BOOST_TEST_MODULE Kernels_SubrangeKernel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "KernelDerivativeTestHelper.h"
using namespace shark;



//since SubrangeKernel is based on weighted sum kernel (which is properly tested) we don't need to do numerical testing.
BOOST_AUTO_TEST_SUITE (Models_Kernels_SubrangeKernel)

BOOST_AUTO_TEST_CASE( DenseSubrangeKernel_Test)
{
	
	//create Data
	std::size_t const examples = 35;
	std::size_t const dim1 = 10;
	std::size_t const dim2 = 12;
	std::vector<RealVector> data(examples,RealVector(dim1+dim2));
	std::vector<RealVector> dataV1(examples,RealVector(dim1));
	std::vector<RealVector> dataV2(examples,RealVector(dim2));
	for(std::size_t i = 0; i != examples; ++i){		
		for(std::size_t j = 0; j != dim1; ++j){
			dataV1[i](j)=data[i](j)=Rng::uni(-1,1);
		}
		for(std::size_t j = 0; j != dim2; ++j){
			dataV2[i](j)=data[i](j+dim1)=Rng::uni(-1,1);
		}
	}
	Data<RealVector> dataset = createDataFromRange(data,10);
	Data<RealVector> datasetV1 = createDataFromRange(dataV1,10);
	Data<RealVector> datasetV2 = createDataFromRange(dataV2,10);
	
	//create Subrange Kernel
	std::vector<std::pair<std::size_t,std::size_t> > ranges(2);
	ranges[0].first=0; ranges[0].second=dim1;
	ranges[1].first=dim1; ranges[1].second=dim1+dim2;
	
	DenseRbfKernel  baseKernelV1(0.2);
	DenseLinearKernel  baseKernelV2;
	std::vector<AbstractKernelFunction<RealVector>*> kernels;
	kernels.push_back(&baseKernelV1);
	kernels.push_back(&baseKernelV2);
	
	SubrangeKernel<RealVector> kernel(kernels,ranges);
	
	//check correct number of parameters
	const unsigned int numParameters = 2;
	kernel.setAdaptiveAll(true);
	BOOST_REQUIRE_EQUAL(kernel.numberOfParameters(),numParameters);
	
	// test kernel evals. first set weighting factors
	RealVector parameters(2);
	init(parameters)<<0.3,0.5; //weights are 1 and 0.3 and the gauss kernel parameter is 0.5
	kernel.setParameterVector(parameters);
	
	//process kernel matrices for each element separately and weight the results to get ground-truth data
	RealMatrix matV1 = calculateRegularizedKernelMatrix(baseKernelV1,datasetV1);
	RealMatrix matV2 = calculateRegularizedKernelMatrix(baseKernelV2,datasetV2);
	RealMatrix kernelMatTest = matV1+std::exp(0.3)*matV2;
	kernelMatTest /=1+std::exp(0.3);
	
	//now calculate the kernel matrix of the MKL Kernel. it should be the same.
	RealMatrix kernelMat = calculateRegularizedKernelMatrix(kernel,dataset);
	
	//test
	for(std::size_t i = 0; i != examples; ++i){
		for(std::size_t j = 0; j != examples;++j){
			BOOST_CHECK_CLOSE(kernelMatTest(i,j),kernelMat(i,j),1.e-5);
		}
	}
	testEval(kernel,dataset.batch(0),dataset.batch(3));
	testKernelDerivative(kernel,22,1.e-7);
	testKernelInputDerivative(kernel,22);
}
BOOST_AUTO_TEST_SUITE_END()
