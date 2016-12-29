#define BOOST_TEST_MODULE BLAS_Symm_Solve
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Core/Shark.h>
#include <shark/LinAlg/BLAS/solve.hpp>
#include <shark/LinAlg/BLAS/matrix.hpp>
#include <shark/LinAlg/BLAS/matrix_expression.hpp>
#include <shark/LinAlg/BLAS/matrix_proxy.hpp>
#include <shark/LinAlg/BLAS/vector_expression.hpp>
#include <shark/LinAlg/BLAS/io.hpp>

#include <iostream>
#include <boost/mpl/list.hpp>

using namespace shark;

//the matrix is designed such that a lot of permutations will be performed
blas::matrix<double> createSymm(std::size_t dimensions, std::size_t rank = 0){
	if(rank == 0) rank = dimensions;
	blas::matrix<double> R(dimensions,dimensions,0.0);
	
	for(std::size_t i = 0; i != dimensions; ++i){
		for(std::size_t j = 0; j <std::min(i,rank); ++j){
			R(i,j) = 0.2/std::abs((int)i -(int)j);
		}
		if(i < rank)
			R(i,i) = 0.5/dimensions*i+1;
	}
	blas::matrix<double> A = prod(R,trans(R));
	if(rank != dimensions){
		for(std::size_t i = 0; i != rank/2; ++i){
			A.swap_rows(2*i,dimensions-i-1);
			A.swap_columns(2*i,dimensions-i-1);
		}
	}
	return A;
}
//simple test which checks for all argument combinations whether they are correctly translated
BOOST_AUTO_TEST_SUITE (Solve_Symm)

BOOST_AUTO_TEST_CASE( Solve_Symm_Vector ){
	std::size_t Dimensions = 128;
	blas::matrix<double> A = createSymm(Dimensions);
	blas::vector<double> b(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		b(i) = (1.0/Dimensions) * i;
	}
	
	//Ax=b
	{
		blas::vector<double> x = blas::solve(A,b, blas::symm_pos_def(),blas::left());
		blas::vector<double> test = prod(A,x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		blas::vector<double> x = blas::solve(trans(A),b, blas::symm_pos_def(),blas::left());
		blas::vector<double> test = prod(trans(A),x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		blas::vector<double> x = blas::solve(A,b, blas::symm_pos_def(),blas::right());
		blas::vector<double> test = prod(x,A);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		blas::vector<double> x = blas::solve(trans(A),b, blas::symm_pos_def(),blas::right());
		blas::vector<double> test = prod(x,trans(A));
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

typedef boost::mpl::list<blas::row_major,blas::column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE(Solve_Matrix, Orientation,result_orientations) {
	std::size_t Dimensions = 128;
	std::size_t k = 151;
	
	std::cout<<"blas::solve Symmetric matrix"<<std::endl;
	blas::matrix<double> A = createSymm(Dimensions);
	blas::matrix<double,Orientation> B(Dimensions,k);
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j < k; ++j){
			B(i,j) = 0.1*j+(1.0/Dimensions) * i;
		}
	}
	blas::matrix<double,Orientation> Bright = trans(B);
	//Ax=b
	{
		blas::matrix<double,Orientation> X= blas::solve(A,B, blas::symm_pos_def(),blas::left());
		blas::matrix<double> test = prod(A,X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		blas::matrix<double,Orientation> X = blas::solve(trans(A),B, blas::symm_pos_def(),blas::left());
		blas::matrix<double> test = prod(trans(A),X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		blas::matrix<double,Orientation> X = blas::solve(A,Bright, blas::symm_pos_def(),blas::right());
		blas::matrix<double> test = prod(X,A);
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		blas::matrix<double,Orientation> X = blas::solve(trans(A),Bright, blas::symm_pos_def(),blas::right());
		blas::matrix<double> test = prod(X,trans(A));
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_CASE( Solve_Symm_Semi_Pos_Def_Vector_Full_Rank ){
	std::size_t Dimensions = 128;
	
	std::cout<<"blas::solve Symmetric semi pos def vector, full rank"<<std::endl;
	blas::matrix<double> A = createSymm(Dimensions);
	blas::vector<double> b(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		b(i) = (1.0/Dimensions) * i;
	}
	
	//Ax=b
	{
		blas::vector<double> x = blas::solve(A,b, blas::symm_semi_pos_def(),blas::left());
		blas::vector<double> test = prod(A,x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		blas::vector<double> x = blas::solve(trans(A),b, blas::symm_semi_pos_def(),blas::left());
		blas::vector<double> test = prod(trans(A),x);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		blas::vector<double> x = blas::solve(A,b, blas::symm_semi_pos_def(),blas::right());
		blas::vector<double> test = prod(x,A);
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		blas::vector<double> x = blas::solve(trans(A),b, blas::symm_semi_pos_def(),blas::right());
		blas::vector<double> test = prod(x,trans(A));
		double error = norm_inf(test-b);
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

typedef boost::mpl::list<blas::row_major,blas::column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE(Solve_Symm_Semi_Pos_Def_Matrix_Full_Rank, Orientation,result_orientations) {
	std::size_t Dimensions = 128;
	std::size_t k = 151;
	
	std::cout<<"blas::solve Symmetric semi pos def matrix, full rank"<<std::endl;
	blas::matrix<double> A = createSymm(Dimensions);
	blas::matrix<double,Orientation> B(Dimensions,k);
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j < k; ++j){
			B(i,j) = 0.1*j+(1.0/Dimensions) * i;
		}
	}
	blas::matrix<double,Orientation> Bright = trans(B);
	//Ax=b
	{
		blas::matrix<double,Orientation> X= blas::solve(A,B, blas::symm_semi_pos_def(),blas::left());
		blas::matrix<double> test = prod(A,X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		blas::matrix<double,Orientation> X = blas::solve(trans(A),B, blas::symm_semi_pos_def(),blas::left());
		blas::matrix<double> test = prod(trans(A),X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		blas::matrix<double,Orientation> X = blas::solve(A,Bright, blas::symm_semi_pos_def(),blas::right());
		blas::matrix<double> test = prod(X,A);
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		blas::matrix<double,Orientation> X = blas::solve(trans(A),Bright, blas::symm_semi_pos_def(),blas::right());
		blas::matrix<double> test = prod(X,trans(A));
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_CASE( Solve_Symm_Semi_Pos_Def_Vector_Rank_Deficient ){
	std::size_t Dimensions = 128;
	std::size_t Rank = 50;
	
	std::cout<<"blas::solve Symmetric semi pos def vector, rank deficient"<<std::endl;
	blas::matrix<double> A = createSymm(Dimensions,Rank);
	blas::vector<double> b(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		b(i) = (1.0/Dimensions) * i;
	}
	
	//if A is not full rank and b i not orthogonal to the nullspace,
	//Ax -b does not hold for the right solution.
	//instead A(Ax-b) must be small (the residual of Ax-b must be orthogonal to A)
	
	//Ax=b
	{
		blas::vector<double> x = blas::solve(A,b, blas::symm_semi_pos_def(),blas::left());
		blas::vector<double> diff = prod(A,x)-b;
		double error = norm_inf(prod(A,diff));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		blas::vector<double> x = blas::solve(trans(A),b, blas::symm_semi_pos_def(),blas::left());
		blas::vector<double> diff = prod(A,x)-b;
		double error = norm_inf(prod(A,diff));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		blas::vector<double> x = blas::solve(A,b, blas::symm_semi_pos_def(),blas::right());
		blas::vector<double> diff = prod(A,x)-b;
		double error = norm_inf(prod(A,diff));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		blas::vector<double> x = blas::solve(trans(A),b, blas::symm_semi_pos_def(),blas::right());
		blas::vector<double> diff = prod(A,x)-b;
		double error = norm_inf(prod(A,diff));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

typedef boost::mpl::list<blas::row_major,blas::column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE(Solve_Symm_Semi_Pos_Def_Matrix_Rank_Deficient, Orientation,result_orientations) {
	std::size_t Dimensions = 128;
	std::size_t Rank = 50;
	std::size_t k = 151;
	
	std::cout<<"blas::solve Symmetric semi pos def matrix, rank deficient"<<std::endl;
	blas::matrix<double> A = createSymm(Dimensions, Rank);
	blas::matrix<double,Orientation> B(Dimensions,k);
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j < k; ++j){
			B(i,j) = 0.1*j+(1.0/Dimensions) * i;
		}
	}
	blas::matrix<double,Orientation> Bright = trans(B);
	//Ax=b
	{
		blas::matrix<double,Orientation> X= blas::solve(A,B, blas::symm_semi_pos_def(),blas::left());
		blas::matrix<double> diff = prod(A,X) - B;
		double error = max(abs(prod(A,diff)));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//A^Tx=b
	{
		blas::matrix<double,Orientation> X = blas::solve(trans(A),B, blas::symm_semi_pos_def(),blas::left());
		blas::matrix<double> diff = prod(A,X) - B;
		double error = max(abs(prod(A,diff)));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	//xA=b
	{
		blas::matrix<double,Orientation> X = blas::solve(A,Bright, blas::symm_semi_pos_def(),blas::right());
		blas::matrix<double> diff = prod(X,A) - Bright;
		double error = max(abs(prod(diff,A)));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
	
	//xA^T=b
	{
		blas::matrix<double,Orientation> X = blas::solve(trans(A),Bright, blas::symm_semi_pos_def(),blas::right());
		blas::matrix<double> diff = prod(X,A) - Bright;
		double error = max(abs(prod(diff,A)));
		BOOST_CHECK_SMALL(error,1.e-12);
	}
}

BOOST_AUTO_TEST_CASE( Solve_Symm_Conjugate_Gradient_Vector ){
	std::size_t Dimensions = 128;
	
	std::cout<<"blas::solve Symmetric conjugate gradient, Vector"<<std::endl;
	blas::matrix<double> A = createSymm(Dimensions);
	blas::vector<double> b(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		b(i) = (1.0/Dimensions) * i;
	}
	
	//if A is not full rank and b i not orthogonal to the nullspace,
	//Ax -b does not hold for the right solution.
	//instead A(Ax-b) must be small (the residual of Ax-b must be orthogonal to A)
	
	//Ax=b
	{
		blas::vector<double> x = blas::solve(A,b, blas::conjugate_gradient(),blas::left(),1.e-9);
		blas::vector<double> diff = prod(A,x)-b;
		double error = norm_inf(diff);
		BOOST_CHECK_SMALL(error,1.e-8);
	}
	//A^Tx=b
	{
		blas::vector<double> x = blas::solve(trans(A),b, blas::conjugate_gradient(),blas::left(),1.e-9);
		blas::vector<double> diff = prod(A,x)-b;
		double error = norm_inf(diff);
		BOOST_CHECK_SMALL(error,1.e-8);
	}
	//xA=b
	{
		blas::vector<double> x = blas::solve(A,b, blas::conjugate_gradient(),blas::right(),1.e-9);
		blas::vector<double> diff = prod(A,x)-b;
		double error = norm_inf(diff);
		BOOST_CHECK_SMALL(error,1.e-8);
	}
	
	//xA^T=b
	{
		blas::vector<double> x = blas::solve(trans(A),b, blas::conjugate_gradient(),blas::right(),1.e-9);
		blas::vector<double> diff = prod(A,x)-b;
		double error = norm_inf(diff);
		BOOST_CHECK_SMALL(error,1.e-8);
	}
}

typedef boost::mpl::list<blas::row_major,blas::column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE(Solve_Symm_Conjugate_Gradient, Orientation,result_orientations) {
	std::size_t Dimensions = 128;
	std::size_t k = 151;
	
	std::cout<<"blas::solve Symmetric semi pos def matrix, full rank"<<std::endl;
	blas::matrix<double> A = createSymm(Dimensions);
	blas::matrix<double,Orientation> B(Dimensions,k);
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j < k; ++j){
			B(i,j) = 0.1*j+(1.0/Dimensions) * i;
		}
	}
	blas::matrix<double,Orientation> Bright = trans(B);
	//Ax=b
	{
		blas::matrix<double,Orientation> X= blas::solve(A,B, blas::conjugate_gradient(),blas::left(),1.e-9);
		blas::matrix<double> test = prod(A,X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-8);
	}
	//A^Tx=b
	{
		blas::matrix<double,Orientation> X = blas::solve(trans(A),B, blas::conjugate_gradient(),blas::left(),1.e-9);
		blas::matrix<double> test = prod(trans(A),X);
		double error = max(abs(test-B));
		BOOST_CHECK_SMALL(error,1.e-8);
	}
	//xA=b
	{
		blas::matrix<double,Orientation> X = blas::solve(A,Bright, blas::conjugate_gradient(),blas::right(),1.e-9);
		blas::matrix<double> test = prod(X,A);
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-8);
	}
	
	//xA^T=b
	{
		blas::matrix<double,Orientation> X = blas::solve(trans(A),Bright, blas::conjugate_gradient(),blas::right(),1.e-9);
		blas::matrix<double> test = prod(X,trans(A));
		double error = max(abs(test-Bright));
		BOOST_CHECK_SMALL(error,1.e-8);
	}
}

BOOST_AUTO_TEST_SUITE_END()
