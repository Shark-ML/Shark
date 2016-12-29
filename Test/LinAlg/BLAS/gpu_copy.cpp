#define BOOST_TEST_MODULE LinAlg_BLAS_GPU_COPY
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Core/Shark.h>
#include <shark/LinAlg/BLAS/blas.h>
#include <shark/LinAlg/BLAS/gpu/vector.hpp>
#include <shark/LinAlg/BLAS/gpu/matrix.hpp>
#include <shark/LinAlg/BLAS/gpu/copy.hpp>

using namespace shark;
using namespace blas;


BOOST_AUTO_TEST_SUITE (LinAlg_BLAS_gpu_copy)

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Vector_Copy ){
	std::cout<<"testing vector copy to gpu and back"<<std::endl;
	vector<float> source(100);
	for(std::size_t i = 0; i != 100; ++i){
		source(i) = 2*i+1;
	}
	gpu::vector<float> target_gpu = gpu::copy_to_gpu(source);
	vector<float> target_cpu = copy_to_cpu(target_gpu);
	
	BOOST_CHECK_SMALL(norm_inf(source - target_cpu), 1.e-10f);	
}
BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Vector_Copy_Plus_Assign ){
	std::cout<<"testing vector assignment to gpu and back"<<std::endl;
	vector<float> source(100);
	for(std::size_t i = 0; i != 100; ++i){
		source(i) = 2*i+1;
	}
	gpu::vector<float> target_gpu(100,1.0);
	noalias(target_gpu) += gpu::copy_to_gpu(source);
	vector<float> target_cpu(100,-2.0);
	noalias(target_cpu) += copy_to_cpu(target_gpu);
	
	BOOST_CHECK_SMALL(norm_inf(source - target_cpu-1), 1.e-10f);	
}

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Matrix_Copy ){
	std::cout<<"testing matrix copy to gpu and back"<<std::endl;
	matrix<float,row_major> source(32,16);
	for(std::size_t i = 0; i != 32; ++i){
		for(std::size_t j = 0; j != 16; ++j){
			source(i,j) = i*16+j;
		}
	}
	matrix<float,column_major> source_cm  = source;
	//row-major cpu to row-major gpu to row-major cpu
	{
		gpu::matrix<float,row_major> target_gpu = gpu::copy_to_gpu(source);
		matrix<float,row_major> target_cpu = copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu), 1.e-10f);
	}
	//row-major-cpu to column-major gpu to column-major cpu
	{
		gpu::matrix<float,column_major> target_gpu = gpu::copy_to_gpu(source);
		matrix<float,column_major> target_cpu = copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu), 1.e-10f);
	}
	//column-major-cpu to column-major gpu to row-major cpu
	{
		gpu::matrix<float,column_major> target_gpu = gpu::copy_to_gpu(source_cm);
		matrix<float,row_major> target_cpu = copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu), 1.e-10f);
	}
	//column-major-cpu to row-major gpu to column-major cpu
	{
		gpu::matrix<float,row_major> target_gpu = gpu::copy_to_gpu(source_cm);
		matrix<float,column_major> target_cpu = copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu), 1.e-10f);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Matrix_Copy_Plus_Assign ){
	std::cout<<"testing matrix assignment to gpu and back"<<std::endl;
	matrix<float,row_major> source(32,16);
	for(std::size_t i = 0; i != 32; ++i){
		for(std::size_t j = 0; j != 16; ++j){
			source(i,j) = i*16+j;
		}
	}
	matrix<float,column_major> source_cm  = source;
	//row-major cpu to row-major gpu to row-major cpu
	{
		gpu::matrix<float,row_major> target_gpu(32,16,1.0);
		noalias(target_gpu) += gpu::copy_to_gpu(source);
		matrix<float,row_major> target_cpu(32,16,-2.0);
		noalias(target_cpu) += copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu-1), 1.e-10f);	
	}
	//row-major-cpu to column-major gpu to column-major cpu
	{
		gpu::matrix<float,column_major> target_gpu(32,16,1.0);
		noalias(target_gpu) += gpu::copy_to_gpu(source);
		matrix<float,column_major> target_cpu(32,16,-2.0);
		noalias(target_cpu) += copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu-1), 1.e-10f);	
	}
	//column-major-cpu to column-major gpu to row-major cpu
	{
		gpu::matrix<float,column_major> target_gpu(32,16,1.0);
		noalias(target_gpu) += gpu::copy_to_gpu(source_cm);
		matrix<float,row_major> target_cpu(32,16,-2.0);
		noalias(target_cpu) += copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu-1), 1.e-10f);	
	}
	//column-major-cpu to row-major gpu to column-major cpu
	{
		gpu::matrix<float,row_major> target_gpu(32,16,1.0);
		noalias(target_gpu) += gpu::copy_to_gpu(source_cm);
		matrix<float,column_major> target_cpu(32,16,-2.0);
		noalias(target_cpu) += copy_to_cpu(target_gpu);
		BOOST_CHECK_SMALL(norm_inf(source - target_cpu-1), 1.e-10f);	
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
