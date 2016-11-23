#define BOOST_TEST_MODULE BLAS_Solve_Triangular
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/solve.hpp>
#include <shark/LinAlg/BLAS/matrix.hpp>
#include <shark/LinAlg/BLAS/matrix_expression.hpp>
#include <shark/LinAlg/BLAS/vector_expression.hpp>
#include <shark/LinAlg/BLAS/io.hpp>

#include <iostream>

using namespace shark;

//simple test which checks for all argument combinations whether they are correctly translated
BOOST_AUTO_TEST_SUITE (Solve_Triangular)

BOOST_AUTO_TEST_CASE( Solve_Vector ){
	std::size_t size = 158;
	
	blas::matrix<double,blas::row_major> A(size,size,1.0);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j <=i; ++j){
			A(i,j) = 0.1/size*i-0.05/(i+1.0)*j;
			if(i ==j)
				A(i,j) += 0.5;
		}
	}
	blas::matrix<double,blas::row_major> Aupper = trans(A);
	
	blas::vector<double> b(size);
	for(std::size_t i = 0; i != size; ++i){
		b(i) = (0.1/size)*i;
	}

	std::cout<<"triangular vector"<<std::endl;
	std::cout<<"left - lower"<<std::endl;
	{
		blas::vector<double> testResult = b;
		blas::solve(A,testResult,blas::left(), blas::lower());
		blas::vector<double> result = blas::triangular_prod<blas::lower>(A,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"right - lower"<<std::endl;
	{
		blas::vector<double> testResult = b;
		blas::solve(A,testResult,blas::right(), blas::lower());
		blas::vector<double> result = blas::triangular_prod<blas::upper>(Aupper,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	
	std::cout<<"left - unit_lower"<<std::endl;
	{
		blas::vector<double> testResult = b;
		blas::solve(A,testResult,blas::left(), blas::unit_lower());
		blas::vector<double> result = blas::triangular_prod<blas::unit_lower>(A,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"right - unit_lower"<<std::endl;
	{
		blas::vector<double> testResult = b;
		blas::solve(A,testResult,blas::right(), blas::unit_lower());
		blas::vector<double> result = blas::triangular_prod<blas::unit_upper>(Aupper,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	
	std::cout<<"left - upper"<<std::endl;
	{
		blas::vector<double> testResult = b;
		blas::solve(Aupper,testResult,blas::left(), blas::upper());
		blas::vector<double> result = blas::triangular_prod<blas::upper>(Aupper,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"right - upper"<<std::endl;
	{
		blas::vector<double> testResult = b;
		blas::solve(Aupper,testResult,blas::right(), blas::upper());
		blas::vector<double> result = blas::triangular_prod<blas::lower>(A,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	
	std::cout<<"left - unit_upper"<<std::endl;
	{
		blas::vector<double> testResult = b;
		blas::solve(Aupper,testResult,blas::left(), blas::unit_upper());
		blas::vector<double> result = blas::triangular_prod<blas::unit_upper>(Aupper,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"right - unit_upper"<<std::endl;
	{
		blas::vector<double> testResult = b;
		blas::solve(Aupper,testResult,blas::right(), blas::unit_upper());
		blas::vector<double> result = blas::triangular_prod<blas::unit_lower>(A,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
}

BOOST_AUTO_TEST_CASE( Solve_Matrix ){
	std::size_t size = 158;
	std::size_t k = 138;
	
	blas::matrix<double,blas::row_major> A(size,size,1.0);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j <=i; ++j){
			A(i,j) = 0.1/size*i-0.05/(i+1.0)*j;
			if(i ==j)
				A(i,j) += 0.5;
		}
	}
	blas::matrix<double,blas::row_major> Aupper = trans(A);
	
	blas::matrix<double,blas::row_major> B(size,k);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j != k; ++j){
			B(i,j) = (0.1/size)*i+0.1*k;
		}
	}
	blas::matrix<double,blas::row_major> Bright = trans(B);

	std::cout<<"triangular matrix"<<std::endl;
	std::cout<<"left - lower - row major"<<std::endl;
	{
		blas::matrix<double,blas::row_major> testResult = B;
		blas::solve(A,testResult,blas::left(), blas::lower());
		blas::matrix<double,blas::row_major> result = blas::triangular_prod<blas::lower>(A,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"right - lower - row major"<<std::endl;
	{
		blas::matrix<double,blas::row_major> testResult = Bright;
		blas::solve(A,testResult,blas::right(), blas::lower());
		blas::matrix<double> result = trans(blas::matrix<double>(blas::triangular_prod<blas::upper>(Aupper,trans(testResult))));
		double error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"left - unit_lower - row major"<<std::endl;
	{
		blas::matrix<double,blas::row_major> testResult = B;
		blas::solve(A,testResult,blas::left(), blas::unit_lower());
		blas::matrix<double,blas::row_major> result = blas::triangular_prod<blas::unit_lower>(A,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"right - unit_lower - row major"<<std::endl;
	{
		blas::matrix<double,blas::row_major> testResult = Bright;
		blas::solve(A,testResult,blas::right(), blas::unit_lower());
		blas::matrix<double,blas::row_major> result = trans(blas::matrix<double>(blas::triangular_prod<blas::unit_upper>(Aupper,trans(testResult))));
		double error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	
	std::cout<<"left - upper - row major"<<std::endl;
	{
		blas::matrix<double,blas::row_major> testResult = B;
		blas::solve(Aupper,testResult,blas::left(), blas::upper());
		blas::matrix<double,blas::row_major> result = blas::triangular_prod<blas::upper>(Aupper,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"right - upper - row major"<<std::endl;
	{
		blas::matrix<double,blas::row_major> testResult = Bright;
		blas::solve(Aupper,testResult,blas::right(), blas::upper());
		blas::matrix<double> result = trans(blas::matrix<double>(blas::triangular_prod<blas::lower>(A,trans(testResult))));
		double error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"left - unit_upper - row major"<<std::endl;
	{
		blas::matrix<double,blas::row_major> testResult = B;
		blas::solve(Aupper,testResult,blas::left(), blas::unit_upper());
		blas::matrix<double,blas::row_major> result = blas::triangular_prod<blas::unit_upper>(Aupper,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"right - unit_upper - row major"<<std::endl;
	{
		blas::matrix<double,blas::row_major> testResult = Bright;
		blas::solve(Aupper,testResult,blas::right(), blas::unit_upper());
		blas::matrix<double,blas::row_major> result = trans(blas::matrix<double>(blas::triangular_prod<blas::unit_lower>(A,trans(testResult))));
		double error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	
	
	std::cout<<"left - lower - column major"<<std::endl;
	{
		blas::matrix<double,blas::column_major> testResult = B;
		blas::solve(A,testResult,blas::left(), blas::lower());
		blas::matrix<double,blas::row_major> result = blas::triangular_prod<blas::lower>(A,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"right - lower - column major"<<std::endl;
	{
		blas::matrix<double,blas::column_major> testResult = Bright;
		blas::solve(A,testResult,blas::right(), blas::lower());
		blas::matrix<double> result = trans(blas::matrix<double>(blas::triangular_prod<blas::upper>(Aupper,trans(testResult))));
		double error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"left - unit_lower - column major"<<std::endl;
	{
		blas::matrix<double,blas::column_major> testResult = B;
		blas::solve(A,testResult,blas::left(), blas::unit_lower());
		blas::matrix<double,blas::row_major> result = blas::triangular_prod<blas::unit_lower>(A,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"right - unit_lower - column major"<<std::endl;
	{
		blas::matrix<double,blas::column_major> testResult = Bright;
		blas::solve(A,testResult,blas::right(), blas::unit_lower());
		blas::matrix<double,blas::row_major> result = trans(blas::matrix<double>(blas::triangular_prod<blas::unit_upper>(Aupper,trans(testResult))));
		double error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	
	std::cout<<"left - upper - column major"<<std::endl;
	{
		blas::matrix<double,blas::column_major> testResult = B;
		blas::solve(Aupper,testResult,blas::left(), blas::upper());
		blas::matrix<double,blas::row_major> result = blas::triangular_prod<blas::upper>(Aupper,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"right - upper - column major"<<std::endl;
	{
		blas::matrix<double,blas::column_major> testResult = Bright;
		blas::solve(Aupper,testResult,blas::right(), blas::upper());
		blas::matrix<double> result = trans(blas::matrix<double>(blas::triangular_prod<blas::lower>(A,trans(testResult))));
		double error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"left - unit_upper - column major"<<std::endl;
	{
		blas::matrix<double,blas::column_major> testResult = B;
		blas::solve(Aupper,testResult,blas::left(), blas::unit_upper());
		blas::matrix<double,blas::row_major> result = blas::triangular_prod<blas::unit_upper>(Aupper,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
	std::cout<<"right - unit_upper - column major"<<std::endl;
	{
		blas::matrix<double,blas::column_major> testResult = Bright;
		blas::solve(Aupper,testResult,blas::right(), blas::unit_upper());
		blas::matrix<double,blas::row_major> result = trans(blas::matrix<double>(blas::triangular_prod<blas::unit_lower>(A,trans(testResult))));
		double error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-12);
	}
}

BOOST_AUTO_TEST_SUITE_END()
