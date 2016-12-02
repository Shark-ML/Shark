#define BOOST_TEST_MODULE BLAS_GPU_Solve_Triangular
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#define SHARK_USE_CLBLAS
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

BOOST_AUTO_TEST_CASE( GPU_Solve_matrix ){
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
	std::cout<<"left - lower - row major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::row_major> testResult = B;
		blas::solve(A,testResult,blas::left(), blas::lower());
		blas::gpu::matrix<float,blas::row_major> result = blas::triangular_prod<blas::lower>(A,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - lower - row major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::row_major> testResult = Bright;
		blas::solve(A,testResult,blas::right(), blas::lower());
		blas::gpu::matrix<float> result = trans(blas::gpu::matrix<float>(blas::triangular_prod<blas::upper>(Aupper,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"left - unit_lower - row major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::row_major> testResult = B;
		blas::solve(A,testResult,blas::left(), blas::unit_lower());
		blas::gpu::matrix<float,blas::row_major> result = blas::triangular_prod<blas::unit_lower>(A,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_lower - row major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::row_major> testResult = Bright;
		blas::solve(A,testResult,blas::right(), blas::unit_lower());
		blas::gpu::matrix<float,blas::row_major> result = trans(blas::gpu::matrix<float>(blas::triangular_prod<blas::unit_upper>(Aupper,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - upper - row major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::row_major> testResult = B;
		blas::solve(Aupper,testResult,blas::left(), blas::upper());
		blas::gpu::matrix<float,blas::row_major> result = blas::triangular_prod<blas::upper>(Aupper,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - upper - row major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::row_major> testResult = Bright;
		blas::solve(Aupper,testResult,blas::right(), blas::upper());
		blas::gpu::matrix<float> result = trans(blas::gpu::matrix<float>(blas::triangular_prod<blas::lower>(A,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"left - unit_upper - row major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::row_major> testResult = B;
		blas::solve(Aupper,testResult,blas::left(), blas::unit_upper());
		blas::gpu::matrix<float,blas::row_major> result = blas::triangular_prod<blas::unit_upper>(Aupper,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_upper - row major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::row_major> testResult = Bright;
		blas::solve(Aupper,testResult,blas::right(), blas::unit_upper());
		blas::gpu::matrix<float,blas::row_major> result = trans(blas::gpu::matrix<float>(blas::triangular_prod<blas::unit_lower>(A,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	
	std::cout<<"left - lower - column major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::column_major> testResult = B;
		blas::solve(A,testResult,blas::left(), blas::lower());
		blas::gpu::matrix<float,blas::row_major> result = blas::triangular_prod<blas::lower>(A,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - lower - column major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::column_major> testResult = Bright;
		blas::solve(A,testResult,blas::right(), blas::lower());
		blas::gpu::matrix<float> result = trans(blas::gpu::matrix<float>(blas::triangular_prod<blas::upper>(Aupper,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"left - unit_lower - column major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::column_major> testResult = B;
		blas::solve(A,testResult,blas::left(), blas::unit_lower());
		blas::gpu::matrix<float,blas::row_major> result = blas::triangular_prod<blas::unit_lower>(A,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_lower - column major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::column_major> testResult = Bright;
		blas::solve(A,testResult,blas::right(), blas::unit_lower());
		blas::gpu::matrix<float,blas::row_major> result = trans(blas::gpu::matrix<float>(blas::triangular_prod<blas::unit_upper>(Aupper,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	
	std::cout<<"left - upper - column major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::column_major> testResult = B;
		blas::solve(Aupper,testResult,blas::left(), blas::upper());
		blas::gpu::matrix<float,blas::row_major> result = blas::triangular_prod<blas::upper>(Aupper,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - upper - column major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::column_major> testResult = Bright;
		blas::solve(Aupper,testResult,blas::right(), blas::upper());
		blas::gpu::matrix<float> result = trans(blas::gpu::matrix<float>(blas::triangular_prod<blas::lower>(A,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"left - unit_upper - column major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::column_major> testResult = B;
		blas::solve(Aupper,testResult,blas::left(), blas::unit_upper());
		blas::gpu::matrix<float,blas::row_major> result = blas::triangular_prod<blas::unit_upper>(Aupper,testResult);
		float error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
	std::cout<<"right - unit_upper - column major"<<std::endl;
	{
		blas::gpu::matrix<float,blas::column_major> testResult = Bright;
		blas::solve(Aupper,testResult,blas::right(), blas::unit_upper());
		blas::gpu::matrix<float,blas::row_major> result = trans(blas::gpu::matrix<float>(blas::triangular_prod<blas::unit_lower>(A,trans(testResult))));
		float error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-5f);
	}
}

BOOST_AUTO_TEST_SUITE_END()
