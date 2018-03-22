#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <shark/Core/Images/Reorder.h>
#include <shark/Core/Random.h>

#define BOOST_TEST_MODULE Core_ImageReorder
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;
using namespace std;

struct ImageReorderFixture {
	ImageReorderFixture(): sizesIn({10,20,7,15}) {
		totalSize = sizesIn[0] *  sizesIn[1] *  sizesIn[2] *  sizesIn[3];
		values_CHWN.resize(totalSize);
		values_NCWH.resize(totalSize);
		values_NHWC = blas::normal(random::globalRng, totalSize, 0, 1, blas::cpu_tag()); 
		
		double (*valuesIn)[10][20][7][15] = (double (*)[10][20][7][15]) values_NHWC.raw_storage().values;
		double (*valuesCHWN)[15][20][7][10]= (double (*)[15][20][7][10]) values_CHWN.raw_storage().values;
		double (*valuesNCWH)[10][15][7][20] = (double (*)[10][15][7][20]) values_NCWH.raw_storage().values;
		for(std::size_t i0 = 0; i0 != sizesIn[0]; ++i0){
			for(std::size_t i1 = 0; i1 != sizesIn[1]; ++i1){
				for(std::size_t i2 = 0; i2 != sizesIn[2]; ++i2){
					for(std::size_t i3 = 0; i3 != sizesIn[3]; ++i3){
						(*valuesCHWN)[i3][i1][i2][i0] = (*valuesIn)[i0][i1][i2][i3];
						(*valuesNCWH)[i0][i3][i2][i1] = (*valuesIn)[i0][i1][i2][i3];
					}
				}
			}
		}
	}

	Shape sizesIn;
	std::size_t totalSize;
	RealVector values_NHWC; 
	RealVector values_CHWN;
	RealVector values_NCWH;
	
};


BOOST_FIXTURE_TEST_SUITE (Core_ImageReorder_Tests, ImageReorderFixture )

BOOST_AUTO_TEST_CASE( Core_Reorder_NHWC_TO_CHWN_CPU){
	RealVector values_test(totalSize,0.0);
	image::reorder<double, blas::cpu_tag>(values_NHWC, values_test, sizesIn, ImageFormat::NHWC, ImageFormat::CHWN);
	BOOST_CHECK_SMALL(norm_inf(values_test - values_CHWN), 1.e-10);
}
BOOST_AUTO_TEST_CASE( Core_Reorder_NHWC_TO_NCWH_CPU){
	RealVector values_test(totalSize,0.0);
	image::reorder<double, blas::cpu_tag>(values_NHWC, values_test, sizesIn, ImageFormat::NHWC, ImageFormat::NCWH);
	BOOST_CHECK_SMALL(norm_inf(values_test - values_NCWH), 1.e-10);
}
BOOST_AUTO_TEST_CASE( Core_Reorder_NCWH_TO_CHWN_CPU){
	RealVector values_test(totalSize,0.0);
	image::reorder<double, blas::cpu_tag>(values_NCWH, values_test, {10, 15, 7, 20}, ImageFormat::NCWH, ImageFormat::CHWN);
	BOOST_CHECK_SMALL(norm_inf(values_test - values_CHWN), 1.e-10);
}
#ifdef SHARK_USE_OPENCL
BOOST_AUTO_TEST_CASE( Core_Reorder_NHWC_TO_CHWN_GPU){
	FloatGPUVector values_test_gpu(totalSize,0.0);
	FloatGPUVector input_gpu = blas::copy_to_gpu(values_NHWC);
	image::reorder<float, blas::gpu_tag>(input_gpu, values_test_gpu, sizesIn, ImageFormat::NHWC, ImageFormat::CHWN);
	FloatVector values_test = blas::copy_to_cpu(values_test_gpu);
	FloatVector ground_truth = values_CHWN;
	BOOST_CHECK_SMALL(norm_inf(values_test - ground_truth), 1.e-10f);
}
BOOST_AUTO_TEST_CASE( Core_Reorder_NHWC_TO_NCWH_GPU){
	FloatGPUVector values_test_gpu(totalSize,0.0);
	FloatGPUVector input_gpu = blas::copy_to_gpu(values_NHWC);
	image::reorder<float, blas::gpu_tag>(input_gpu, values_test_gpu, sizesIn, ImageFormat::NHWC, ImageFormat::NCWH);
	FloatVector values_test = blas::copy_to_cpu(values_test_gpu);
	FloatVector ground_truth = values_NCWH;
	BOOST_CHECK_SMALL(norm_inf(values_test - ground_truth), 1.e-10f);
}
BOOST_AUTO_TEST_CASE( Core_Reorder_NCWH_TO_CHWN_GPU){
	FloatGPUVector values_test_gpu(totalSize,0.0);
	FloatGPUVector input_gpu = blas::copy_to_gpu(values_NCWH);
	image::reorder<float, blas::gpu_tag>(input_gpu, values_test_gpu, {10, 15, 7, 20}, ImageFormat::NCWH, ImageFormat::CHWN);
	FloatVector values_test = blas::copy_to_cpu(values_test_gpu);
	FloatVector ground_truth = values_CHWN;
	BOOST_CHECK_SMALL(norm_inf(values_test - ground_truth), 1.e-10f);
}
#endif
BOOST_AUTO_TEST_SUITE_END()
