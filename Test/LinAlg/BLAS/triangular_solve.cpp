#define BOOST_TEST_MODULE BLAS_Solve_Triangular
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Core/Shark.h>
#include <shark/LinAlg/BLAS/solve.hpp>
#include <shark/LinAlg/BLAS/matrix.hpp>
#include <shark/LinAlg/BLAS/matrix_expression.hpp>
#include <shark/LinAlg/BLAS/matrix_proxy.hpp>
#include <shark/LinAlg/BLAS/vector_expression.hpp>

#include <iostream>
#include <boost/mpl/list.hpp>

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
		blas::vector<double> testResult = solve(A,b, blas::lower(), blas::left());
		blas::vector<double> resultProd = prod(inv(A,blas::lower()),b);
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);//check that both expressions are the same
		blas::vector<double> result = blas::triangular_prod<blas::lower>(A,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - lower"<<std::endl;
	{
		blas::vector<double> testResult = solve(A,b, blas::lower(), blas::right());
		blas::vector<double> resultProd = prod(b,inv(A,blas::lower()));
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		blas::vector<double> result = blas::triangular_prod<blas::upper>(Aupper,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	
	std::cout<<"left - unit_lower"<<std::endl;
	{
		blas::vector<double> testResult = solve(A,b,blas::unit_lower(), blas::left());
		blas::vector<double> resultProd = prod(inv(A,blas::unit_lower()),b);
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		blas::vector<double> result = blas::triangular_prod<blas::unit_lower>(A,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - unit_lower"<<std::endl;
	{
		blas::vector<double> testResult = solve(A,b, blas::unit_lower(), blas::right());
		blas::vector<double> resultProd = prod(b,inv(A,blas::unit_lower()));
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		blas::vector<double> result = blas::triangular_prod<blas::unit_upper>(Aupper,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	
	std::cout<<"left - upper"<<std::endl;
	{
		blas::vector<double> testResult = solve(Aupper,b, blas::upper(), blas::left());
		blas::vector<double> resultProd = prod(inv(Aupper,blas::upper()),b);
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		blas::vector<double> result = blas::triangular_prod<blas::upper>(Aupper,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - upper"<<std::endl;
	{
		blas::vector<double> testResult = solve(Aupper,b,blas::upper(), blas::right());
		blas::vector<double> resultProd = prod(b,inv(Aupper,blas::upper()));
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		blas::vector<double> result = blas::triangular_prod<blas::lower>(A,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	
	std::cout<<"left - unit_upper"<<std::endl;
	{
		blas::vector<double> testResult = solve(Aupper,b, blas::unit_upper(), blas::left());
		blas::vector<double> resultProd = prod(inv(Aupper,blas::unit_upper()),b);
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		blas::vector<double> result = blas::triangular_prod<blas::unit_upper>(Aupper,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - unit_upper"<<std::endl;
	{
		blas::vector<double> testResult = solve(Aupper,b, blas::unit_upper(), blas::right());
		blas::vector<double> resultProd = prod(b,inv(Aupper,blas::unit_upper()));
		BOOST_CHECK_SMALL(norm_inf(testResult - resultProd),1.e-15);
		blas::vector<double> result = blas::triangular_prod<blas::unit_lower>(A,testResult);
		double error = norm_inf(result-b);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
}

typedef boost::mpl::list<blas::row_major,blas::column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE(Solve_Matrix, Orientation,result_orientations) {
	std::size_t size = 158;
	std::size_t k = 138;
	
	blas::matrix<double,blas::row_major> A(size,size,1.0);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j <=i; ++j){
			A(i,j) = 0.2/size*i-0.05/(i+1.0)*j;
			if(i ==j)
				A(i,j) += 10;
		}
	}
	blas::matrix<double,blas::row_major> Aupper = trans(A);
	
	blas::matrix<double,Orientation> B(size,k);
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j != k; ++j){
			B(i,j) = (0.1/size)*i+0.1*k;
		}
	}
	blas::matrix<double,Orientation> Bright = trans(B);

	std::cout<<"triangular matrix"<<std::endl;
	std::cout<<"left - lower"<<std::endl;
	{
		blas::matrix<double,Orientation> testResult = solve(A,B, blas::lower(), blas::left());
		blas::matrix<double,Orientation> prodResult = prod(inv(A,blas::lower()),B);
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		blas::matrix<double,blas::row_major> result = blas::triangular_prod<blas::lower>(A,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - lower"<<std::endl;
	{
		blas::matrix<double,Orientation> testResult = solve(A,Bright, blas::lower(), blas::right());
		blas::matrix<double,Orientation> prodResult = prod(Bright,inv(A,blas::lower()));
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		blas::matrix<double> result = trans(blas::matrix<double>(blas::triangular_prod<blas::upper>(Aupper,trans(testResult))));
		double error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"left - unit_lower"<<std::endl;
	{
		blas::matrix<double,Orientation> testResult = solve(A,B, blas::unit_lower(), blas::left());
		blas::matrix<double,Orientation> prodResult = prod(inv(A,blas::unit_lower()),B);
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		blas::matrix<double,blas::row_major> result = blas::triangular_prod<blas::unit_lower>(A,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - unit_lower"<<std::endl;
	{
		blas::matrix<double,Orientation> testResult = solve(A,Bright, blas::unit_lower(), blas::right());
		blas::matrix<double,Orientation> prodResult = prod(Bright,inv(A,blas::unit_lower()));
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		blas::matrix<double,blas::row_major> result = trans(blas::matrix<double>(blas::triangular_prod<blas::unit_upper>(Aupper,trans(testResult))));
		double error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	
	std::cout<<"left - upper"<<std::endl;
	{
		blas::matrix<double,Orientation> testResult = solve(Aupper,B, blas::upper(), blas::left());
		blas::matrix<double,Orientation> prodResult = prod(inv(Aupper,blas::upper()),B);
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		blas::matrix<double,blas::row_major> result = blas::triangular_prod<blas::upper>(Aupper,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - upper"<<std::endl;
	{
		blas::matrix<double,Orientation> testResult = solve(Aupper,Bright, blas::upper(), blas::right());
		blas::matrix<double,Orientation> prodResult = prod(Bright,inv(Aupper,blas::upper()));
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		blas::matrix<double> result = trans(blas::matrix<double>(blas::triangular_prod<blas::lower>(A,trans(testResult))));
		double error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"left - unit_upper"<<std::endl;
	{
		blas::matrix<double,Orientation> testResult = solve(Aupper,B, blas::unit_upper(), blas::left());
		blas::matrix<double,Orientation> prodResult = prod(inv(Aupper,blas::unit_upper()),B);
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		blas::matrix<double,blas::row_major> result = blas::triangular_prod<blas::unit_upper>(Aupper,testResult);
		double error = norm_inf(result-B);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
	std::cout<<"right - unit_upper"<<std::endl;
	{
		blas::matrix<double,Orientation> testResult = solve(Aupper,Bright, blas::unit_upper(), blas::right());
		blas::matrix<double,Orientation> prodResult = prod(Bright,inv(Aupper,blas::unit_upper()));
		BOOST_CHECK_SMALL(max(abs(testResult - prodResult)),1.e-15);
		blas::matrix<double,blas::row_major> result = trans(blas::matrix<double>(blas::triangular_prod<blas::unit_lower>(A,trans(testResult))));
		double error = norm_inf(result-Bright);
		BOOST_CHECK_SMALL(error, 1.e-11);
	}
}

BOOST_AUTO_TEST_SUITE_END()
