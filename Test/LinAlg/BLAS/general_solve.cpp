#define BOOST_TEST_MODULE BLAS_General_Solve
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Core/Shark.h>
#include <shark/LinAlg/BLAS/solve.hpp>
#include <shark/LinAlg/BLAS/vector.hpp>
#include <shark/LinAlg/BLAS/matrix.hpp>
#include <shark/LinAlg/BLAS/matrix_expression.hpp>
#include <shark/LinAlg/BLAS/matrix_proxy.hpp>
#include <shark/LinAlg/BLAS/vector_expression.hpp>

#include <iostream>
#include <boost/mpl/list.hpp>

using namespace shark;

//the matrix is designed such that a lot of permutations will be performed
blas::matrix<double> createMatrix(std::size_t dimensions){
	blas::matrix<double> L(dimensions,dimensions,0.0);
	blas::matrix<double> U(dimensions,dimensions,0.0);
	
	for(std::size_t i = 0; i != dimensions; ++i){
		for(std::size_t j = 0; j <i; ++j){
			U(j,i) = 0.0;//1 - 0.1/dimensions * std::abs((int)i -(int)j);
			L(i,j)  = 3 - 3.0/dimensions*std::abs((int)i -(int)j);
		}
		U(i,i) = 0.5/dimensions*i+1;
		L(i,i) = 1;
	}
	blas::matrix<double> A = prod(L,U);
	diag(A) += 1;
	return A;
}
typedef boost::mpl::list<blas::row_major,blas::column_major> result_orientations;

//simple test which checks for all argument combinations whether they are correctly translated
BOOST_AUTO_TEST_SUITE (Solve_General)

BOOST_AUTO_TEST_CASE( Solve_Indefinite_Full_Rank_Vector ){
	std::size_t Dimensions = 128;
	blas::matrix<double> A = createMatrix(Dimensions);
	blas::vector<double> b(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		b(i) = (1.0/Dimensions) * i;
	}
	
	//Ax=b
	{
		blas::vector<double> x = blas::solve(A,b, blas::indefinite_full_rank(),blas::left());
		blas::vector<double> xprod = prod(inv(A,blas::indefinite_full_rank()),b);
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);
		blas::vector<double> test = prod(A,x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		blas::vector<double> x = blas::solve(trans(A),b, blas::indefinite_full_rank(),blas::left());
		blas::vector<double> xprod = prod(inv(trans(A),blas::indefinite_full_rank()),b);
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);
		blas::vector<double> test = prod(trans(A),x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		blas::vector<double> x = blas::solve(A,b, blas::indefinite_full_rank(),blas::right());
		blas::vector<double> xprod = prod(b,inv(A,blas::indefinite_full_rank()));
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-15);
		blas::vector<double> test = prod(x,A);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		blas::vector<double> x = blas::solve(trans(A),b, blas::indefinite_full_rank(),blas::right());
		blas::vector<double> xprod = prod(b,inv(trans(A),blas::indefinite_full_rank()));
		BOOST_CHECK_SMALL(norm_inf(x-xprod),1.e-14);
		blas::vector<double> test = prod(x,trans(A));
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Solve_Matrix, Orientation,result_orientations) {
	std::size_t Dimensions = 128;
	std::size_t k = 151;
	
	std::cout<<"blas::solve Symmetric matrix"<<std::endl;
	blas::matrix<double> A = createMatrix(Dimensions);
	blas::matrix<double,Orientation> B(Dimensions,k);
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j < k; ++j){
			B(i,j) = 0.1*j+(1.0/Dimensions) * i;
		}
	}
	blas::matrix<double,Orientation> Bright = trans(B);
	//Ax=b
	{
		blas::matrix<double,Orientation> X= blas::solve(A,B, blas::indefinite_full_rank(),blas::left());
		blas::matrix<double,Orientation> Xprod = prod(inv(A,blas::indefinite_full_rank()),B);
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);
		blas::matrix<double> test = prod(A,X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		blas::matrix<double,Orientation> X = blas::solve(trans(A),B, blas::indefinite_full_rank(),blas::left());
		blas::matrix<double,Orientation> Xprod = prod(inv(trans(A),blas::indefinite_full_rank()),B);
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);
		blas::matrix<double> test = prod(trans(A),X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		blas::matrix<double,Orientation> X = blas::solve(A,Bright, blas::indefinite_full_rank(),blas::right());
		blas::matrix<double,Orientation> Xprod = prod(Bright,inv(A,blas::indefinite_full_rank()));
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);
		blas::matrix<double> test = prod(X,A);
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		blas::matrix<double,Orientation> X = blas::solve(trans(A),Bright, blas::indefinite_full_rank(),blas::right());
		blas::matrix<double,Orientation> Xprod = prod(Bright,inv(trans(A),blas::indefinite_full_rank()));
		BOOST_CHECK_SMALL(max(abs(X-Xprod)),1.e-15);
		blas::matrix<double> test = prod(X,trans(A));
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_SUITE_END()
