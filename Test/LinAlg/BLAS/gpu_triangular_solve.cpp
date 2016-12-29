#define BOOST_TEST_MODULE BLAS_GPU_Solve_Triangular
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Core/Shark.h>
#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <shark/LinAlg/BLAS/gpu/vector.hpp>
#include <shark/LinAlg/BLAS/solve.hpp>
#include <shark/LinAlg/BLAS/gpu/matrix.hpp>

#include <shark/LinAlg/BLAS/gpu/copy.hpp>
#include <shark/LinAlg/BLAS/matrix.hpp>
#include <shark/LinAlg/BLAS/matrix_expression.hpp>
#include <shark/LinAlg/BLAS/vector_expression.hpp>

#include <iostream>

using namespace shark;

//simple test which checks for all argument combinations whether they are correctly translated
BOOST_AUTO_TEST_SUITE (GPU_Solve_Triangular)


BOOST_AUTO_TEST_CASE( Solve_Vector ){
	std::size_t size = 326;
	
	blas::matrix<float,blas::row_major> A_cpu(size,size,1.0);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j <=i; ++j){
			A_cpu(i,j) = 0.1/size*i-0.05/(i+1.0)*j;
			if(i ==j)
				A_cpu(i,j) += 10;
		}
	}
	blas::gpu::matrix<float,blas::row_major> A = blas::gpu::copy_to_gpu(A_cpu);
	blas::gpu::matrix<float,blas::row_major> Aupper = trans(A);
	
	blas::vector<float> b_cpu(size);
	for(std::size_t i = 0; i != size; ++i){
		b_cpu(i) = (0.1/size)*i;
	}
	
	blas::gpu::vector<float> b = blas::gpu::copy_to_gpu(b_cpu);

	std::cout<<"triangular vector"<<std::endl;
	std::cout<<"left - lower"<<std::endl;
	{
		blas::gpu::vector<float> testResult = solve(A,b, blas::lower(), blas::left());
		blas::vector<float> result = copy_to_cpu(blas::triangular_prod<blas::lower>(A,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - lower"<<std::endl;
	{
		blas::gpu::vector<float> testResult = solve(A,b, blas::lower(), blas::right());
		blas::vector<float> result = copy_to_cpu(blas::triangular_prod<blas::upper>(Aupper,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - unit_lower"<<std::endl;
	{
		blas::gpu::vector<float> testResult = solve(A,b, blas::unit_ower(), blas::left());
		blas::vector<float> result = copy_to_cpu(blas::triangular_prod<blas::unit_lower>(A,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_lower"<<std::endl;
	{
		blas::gpu::vector<float> testResult = solve(A,b, blas::unit_lower(), blas::right());
		blas::vector<float> result = copy_to_cpu(blas::triangular_prod<blas::unit_upper>(Aupper,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - upper"<<std::endl;
	{
		blas::gpu::vector<float> testResult = solve(Aupper,b, blas::upper(), blas::left());
		blas::vector<float> result = copy_to_cpu(blas::triangular_prod<blas::upper>(Aupper,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - upper"<<std::endl;
	{
		blas::gpu::vector<float> testResult = solve(Aupper,b, blas::upper(), blas::right());
		blas::vector<float> result = copy_to_cpu(blas::triangular_prod<blas::lower>(A,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - unit_upper"<<std::endl;
	{
		blas::gpu::vector<float> testResult = solve(Aupper,b, blas::unit_upper(), blas::left());
		blas::vector<float> result = copy_to_cpu(blas::triangular_prod<blas::unit_upper>(Aupper,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_upper"<<std::endl;
	{
		blas::gpu::vector<float> testResult = solve(Aupper,b, blas::unit_upper(), blas::right());
		blas::vector<float> result = copy_to_cpu(blas::triangular_prod<blas::unit_lower>(A,testResult));
		float error = norm_inf(result-b_cpu);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
}

typedef std::tuple<row_major,column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE(GPU_Solve_Matrix, Orientation,result_orientations) {
	std::size_t size = 139;
	std::size_t k = 238;
	
	blas::matrix<float,blas::row_major> A_cpu(size,size,1.0);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j <=i; ++j){
			A_cpu(i,j) = 0.1/size*i-0.05/(i+1.0)*j;
			if(i ==j)
				A_cpu(i,j) += 10;
		}
	}
	blas::gpu::matrix<float,blas::row_major> A = blas::gpu::copy_to_gpu(A_cpu);
	blas::gpu::matrix<float,blas::row_major> Aupper = trans(A);
	
	
	blas::matrix<float,blas::row_major> B_cpu(size,k);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j != k; ++j){
			B_cpu(i,j) = (0.1/size)*i+0.1/k*j;
		}
	}
	blas::gpu::matrix<float,blas::row_major> B = blas::gpu::copy_to_gpu(B_cpu);
	blas::gpu::matrix<float,blas::row_major> Bright = trans(B);

	std::cout<<"triangular gpu::matrix"<<std::endl;
	std::cout<<"left - lower"<<std::endl;
	{
		blas::gpu::matrix<float,Orientation> testResult = solve(A,B, blas::lower(), blas::left());
		blas::gpu::matrix<float,blas::row_major> result = blas::triangular_prod<blas::lower>(A,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - lower"<<std::endl;
	{
		blas::gpu::matrix<float,Orientation> testResult = solve(A,Bright, blas::lower(), blas::right());
		blas::gpu::matrix<float> result = trans(blas::gpu::matrix<float>(blas::triangular_prod<blas::upper>(Aupper,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"left - unit_lower"<<std::endl;
	{
		blas::gpu::matrix<float,Orientation> testResult = solve(A,B, blas::unit_lower(), blas::left());
		blas::gpu::matrix<float,blas::row_major> result = blas::triangular_prod<blas::unit_lower>(A,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_lower"<<std::endl;
	{
		blas::gpu::matrix<float,Orientation> testResult = solve(A,Bright, blas::unit_lower(), blas::right());
		blas::gpu::matrix<float,blas::row_major> result = trans(blas::gpu::matrix<float>(blas::triangular_prod<blas::unit_upper>(Aupper,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - upper"<<std::endl;
	{
		blas::gpu::matrix<float,Orientation> testResult = solve(A,B, blas::upper(), blas::left());
		blas::gpu::matrix<float,blas::row_major> result = blas::triangular_prod<blas::upper>(Aupper,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - upper"<<std::endl;
	{
		blas::gpu::matrix<float,Orientation> testResult = solve(A,Bright, blas::upper(), blas::right());
		blas::gpu::matrix<float> result = trans(blas::gpu::matrix<float>(blas::triangular_prod<blas::lower>(A,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"left - unit_upper"<<std::endl;
	{
		blas::gpu::matrix<float,Orientation> testResult = solve(A,B, blas::unit_upper(), blas::left());
		blas::gpu::matrix<float,blas::row_major> result = blas::triangular_prod<blas::unit_upper>(Aupper,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_upper"<<std::endl;
	{
		blas::gpu::matrix<float,Orientation> testResult = solve(A,Bright, blas::unit_upper(), blas::right());
		blas::gpu::matrix<float,blas::row_major> result = trans(blas::gpu::matrix<float>(blas::triangular_prod<blas::unit_lower>(A,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
