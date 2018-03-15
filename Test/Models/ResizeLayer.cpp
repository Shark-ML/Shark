#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <iostream>
#include <shark/Models/ResizeLayer.h>
#define BOOST_TEST_MODULE Models_ConvolutionalModel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"


using namespace shark;


BOOST_AUTO_TEST_SUITE (Models_ResizeLayer)

BOOST_AUTO_TEST_CASE( Models_ResizeLayer_Spline_Derivatives){
	Shape imageShape = {3,4,3};
	Shape outputShape = {5,2};
	Shape outputShape3 = {5,2,3};
	ResizeLayer<RealVector> model(imageShape, outputShape, Interpolation::Spline);
	BOOST_REQUIRE_EQUAL(model.numberOfParameters(), 0);
	BOOST_REQUIRE_EQUAL(model.inputShape(), imageShape);
	BOOST_REQUIRE_EQUAL(model.outputShape(), outputShape3);

	testWeightedInputDerivative(model,100,1.e-5, 1.e-5);
}

#ifdef SHARK_USE_OPENCL
BOOST_AUTO_TEST_CASE( Models_ResizeLayer_Spline_GPU){
	
	//define model
	Shape imageShape = {22,41,1};
	Shape outputShape = {25,18,1};
	ResizeLayer<FloatVector> model_cpu(imageShape, outputShape, Interpolation::Spline);
	ResizeLayer<FloatGPUVector> model_gpu(imageShape, outputShape, Interpolation::Spline);
	
	//define inputs
	FloatMatrix images_cpu = blas::uniform(random::globalRng, 1, imageShape.numElements(), float(0), float(1), blas::cpu_tag());
	FloatMatrix coefficients_cpu = blas::uniform(random::globalRng, 1, outputShape.numElements(), float(-1), float(1), blas::cpu_tag());
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
