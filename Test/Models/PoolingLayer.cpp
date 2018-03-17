#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <shark/Models/PoolingLayer.h>
#define BOOST_TEST_MODULE Models_POOLINGLAYER
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"


using namespace shark;


BOOST_AUTO_TEST_SUITE (Models_PoolingLayer)

BOOST_AUTO_TEST_CASE( Models_PoolingLayer_MaxPooling_Value){
	Shape imageShape = {2,4,2};
	Shape patchShape = {2,2};
	Shape outputShape = {1,2,2};
	PoolingLayer<RealVector> model(imageShape, patchShape, Pooling::Maximum);
	
	BOOST_REQUIRE_EQUAL(model.numberOfParameters(), 0);
	BOOST_REQUIRE_EQUAL(model.inputShape(), imageShape);
	BOOST_REQUIRE_EQUAL(model.outputShape(), outputShape);

	RealMatrix dataIn = {
		{1, 10, 2, 22,	-5, -3, -12, 3,
		5, 8, 12, 3, 	-8, 7, -16, 2},
		{5, -3, 1, 3,	1, 10, 2, 22,	
		-5, 7, 1, 2,	5, 8, 12, 3}
	};
	RealMatrix dataOut = {
		{12, 22, -5, 7},
		{5, 7, 12, 22}
	};
	
	RealMatrix test = model(dataIn);
	BOOST_CHECK_SMALL(norm_inf(test - dataOut), 1.e-10);
	
	//create a larger model and test that batch-evaluation is the same
	imageShape = {10,16,3};
	patchShape = {5,2};
	outputShape = {2,8,3};
	PoolingLayer<RealVector> modelBig(imageShape, patchShape, Pooling::Maximum);
	BOOST_REQUIRE_EQUAL(modelBig.numberOfParameters(), 0);
	BOOST_REQUIRE_EQUAL(modelBig.inputShape(), imageShape);
	BOOST_REQUIRE_EQUAL(modelBig.outputShape(), outputShape);
	
	RealMatrix batchInput = blas::normal(random::globalRng, 16, imageShape.numElements(), 0.0, 1.0, blas::cpu_tag());
	testBatchEval(modelBig, batchInput);
}

BOOST_AUTO_TEST_CASE( Models_PoolingLayer_MaxPooling_Derivative){
	Shape imageShape = {10,16,3};
	Shape patchShape = {5,2};
	Shape outputShape = {2,8,3};
	PoolingLayer<RealVector> model(imageShape, patchShape, Pooling::Maximum);
	BOOST_REQUIRE_EQUAL(model.numberOfParameters(), 0);
	BOOST_REQUIRE_EQUAL(model.inputShape(), imageShape);
	BOOST_REQUIRE_EQUAL(model.outputShape(), outputShape);

	testWeightedInputDerivative(model,100,1.e-5, 1.e-8);
	testWeightedDerivativesSame(model, 10);
}

#ifdef SHARK_USE_OPENCL
BOOST_AUTO_TEST_CASE( Models_PoolingLayer_Spline_GPU){
	
	//define model
	Shape imageShape = {24,66,7};
	Shape patchShape = {4,6};
	Shape outputShape = {6,11,7};
	PoolingLayer<FloatVector> model_cpu(imageShape, patchShape, Pooling::Maximum);
	PoolingLayer<FloatGPUVector> model_gpu(imageShape, patchShape, Pooling::Maximum);
	
	//define inputs
	FloatMatrix images_cpu = blas::uniform(random::globalRng, 100, imageShape.numElements(), 0.0f, 1.0f, blas::cpu_tag());
	FloatMatrix coefficients_cpu = blas::uniform(random::globalRng, 100, outputShape.numElements(), 0.0f, 1.0f, blas::cpu_tag());
	FloatGPUMatrix images_gpu = blas::copy_to_gpu(images_cpu);
	FloatGPUMatrix coefficients_gpu = blas::copy_to_gpu(coefficients_cpu);
	
	
	auto state_cpu = model_cpu.createState();
	FloatMatrix result_cpu;
	FloatMatrix gradient_cpu;
	model_cpu.eval(images_cpu, result_cpu, *state_cpu);
	model_cpu.weightedInputDerivative(images_cpu, result_cpu, coefficients_cpu, *state_cpu, gradient_cpu);
	
	auto state_gpu = model_gpu.createState();
	FloatGPUMatrix result_gpu;
	FloatGPUMatrix gradient_gpu;
	model_gpu.eval(images_gpu, result_gpu, *state_gpu);
	model_gpu.weightedInputDerivative(images_gpu, result_gpu, coefficients_gpu, *state_gpu, gradient_gpu);
	
	//check value and derivatives to be the same as cpu version
	FloatMatrix result_gpu_transfered = blas::copy_to_cpu(result_gpu);
	FloatMatrix gradient_gpu_transfered = blas::copy_to_cpu(gradient_gpu);
	BOOST_CHECK_SMALL(norm_inf(result_cpu - result_gpu_transfered), 1.e-5f);
	BOOST_CHECK_SMALL(norm_inf(gradient_cpu - gradient_gpu_transfered), 1.e-5f);
}
#endif
BOOST_AUTO_TEST_SUITE_END()
