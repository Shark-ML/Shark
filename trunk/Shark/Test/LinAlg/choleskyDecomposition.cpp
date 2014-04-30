#define BOOST_TEST_MODULE LinAlg_CholeskyDecomposition
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/Cholesky.h>
#include <shark/LinAlg/rotations.h>

using namespace shark;

const size_t Dimensions=4;
double inputMatrix[Dimensions][Dimensions]=
{
	{ 9,   3, -6,  12},
	{ 3,  26, -7, -11},
	{-6, -7,   9,   7},
	{12, -11,  7,  65}
};

double decomposedMatrix[Dimensions][Dimensions]=
{
	{  3,  0, 0,  0},
	{  1,  5, 0,  0},
	{ -2, -1, 2,  0},
	{  4, -3, 6,  2}
};


BOOST_AUTO_TEST_CASE( LinAlg_CholeskyDecomposition_Base )
{
	RealMatrix M(Dimensions, Dimensions);   // input matrix
	RealMatrix C(Dimensions, Dimensions);   // matrix for Cholesky Decomposition

	// Initializing matrices
	for (size_t row = 0; row < Dimensions; row++)
	{
		for (size_t col = 0; col < Dimensions; col++)
		{
			M(row, col) = inputMatrix[row][col];
			C(row, col) = 0;
		}
	}
	//Decompose
	choleskyDecomposition(M,C);

	//test for equality
	for (size_t row = 0; row < Dimensions; row++)
	{
		for (size_t col = 0; col < Dimensions; col++)
		{
			BOOST_CHECK_SMALL(C(row, col)-decomposedMatrix[row][col],1.e-14);
		}
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_PivotingCholeskyDecomposition_Base )
{
	RealMatrix M(Dimensions, Dimensions);   // input matrix
	RealMatrix C(Dimensions, Dimensions);   // matrix for Cholesky Decomposition
	PermutationMatrix P(Dimensions);
	// Initializing matrices
	for (size_t row = 0; row < Dimensions; row++)
	{
		for (size_t col = 0; col < Dimensions; col++)
		{
			M(row, col) = inputMatrix[row][col];
			C(row, col) = 0;
		}
	}
	//Decompose
	std::size_t rank = pivotingCholeskyDecomposition(M,P,C);
	swap_rows(P,C);
	
	double error = norm_inf(prod(C,trans(C))-M);
	BOOST_CHECK_SMALL(error,1.e-13);
	BOOST_CHECK_EQUAL(rank,4);
}

RealMatrix createRandomMatrix(RealMatrix const& lambda,std::size_t Dimensions){
	RealMatrix R = blas::randomRotationMatrix(Dimensions);
	RealMatrix Atemp(Dimensions,Dimensions);
	RealMatrix A(Dimensions,Dimensions);
	axpy_prod(R,lambda,Atemp);
	axpy_prod(Atemp,trans(R),A);
	return A;
}

BOOST_AUTO_TEST_CASE( LinAlg_CholeskyDecomposition ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 48;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable eigenvalue problem matrix A
		RealMatrix lambda(Dimensions,Dimensions);
		lambda.clear();
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i,i) = Rng::uni(1,3.0);
		}
		RealMatrix A = createRandomMatrix(lambda,Dimensions);
		//calculate Cholesky
		RealMatrix C(Dimensions,Dimensions);
		choleskyDecomposition(A,C);
		
		//test determinant of C
		double logDetA = trace(log(lambda));
		double logDetC = trace(log(sqr(C)));
		BOOST_CHECK_SMALL(std::abs(logDetA)-std::abs(logDetC),1.e-12);

		//create reconstruction of A
		RealMatrix ATest(Dimensions,Dimensions);
		
		axpy_prod(C,trans(C),ATest);
		
		//test reconstruction error
		double errorA = norm_inf(A-ATest);
		BOOST_CHECK_SMALL(errorA,1.e-12);
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(ATest)));//test for nans
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_PivotingCholeskyDecomposition_FullRank ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 48;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable eigenvalue problem matrix A
		RealMatrix lambda(Dimensions,Dimensions);
		lambda.clear();
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i,i) = Rng::uni(1,3.0);
		}
		RealMatrix A = createRandomMatrix(lambda,Dimensions);
		//calculate Cholesky
		RealMatrix C(Dimensions,Dimensions);
		PermutationMatrix P(Dimensions);
		std::size_t rank = pivotingCholeskyDecomposition(A,P,C);
		//test whether result is full rank
		BOOST_CHECK_EQUAL(rank,Dimensions);
		
		//test determinant of C
		double logDetA = trace(log(lambda));
		double logDetC = trace(log(sqr(C)));
		BOOST_CHECK_SMALL(std::abs(logDetA)-std::abs(logDetC),1.e-12);

		//create reconstruction of A
		RealMatrix ATest(Dimensions,Dimensions);
		
		swap_full(P,A);
		
		axpy_prod(C,trans(C),ATest);
		
		//test reconstruction error
		double errorA = norm_inf(A-ATest);
		BOOST_CHECK_SMALL(errorA,1.e-12);
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(ATest)));//test for nans
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_PivotingCholeskyDecomposition_RankK ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 45;
	for(std::size_t test = 0; test != NumTests; ++test){
		std::size_t Rank = Rng::discrete(10,45);
		//first generate a suitable eigenvalue problem matrix A
		RealMatrix lambda(Dimensions,Dimensions);
		lambda.clear();
		for(std::size_t i = 0; i != Rank; ++i){
			lambda(i,i) = Rng::uni(1,3.0);
		}
		RealMatrix A = createRandomMatrix(lambda,Dimensions);
		//calculate Cholesky
		RealMatrix C(Dimensions,Dimensions);
		PermutationMatrix P(Dimensions);
		std::size_t rank = pivotingCholeskyDecomposition(A,P,C);
		
		//test whether result has the correct rank.
		BOOST_CHECK_EQUAL(rank,Rank);

		//create reconstruction of A
		RealMatrix ATest(Dimensions,Dimensions);
		
		swap_full(P,A);
		axpy_prod(C,trans(C),ATest);
		
		//test reconstruction error
		double errorA = norm_inf(A-ATest);
		BOOST_CHECK_SMALL(errorA,1.e-13);
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(ATest)));//test for nans
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_CholeskyUpdate ){
	std::size_t NumTests = 100;
	std::size_t Dimensions = 50;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable eigenvalue problem matrix A as well as its decompisition
		RealMatrix lambda(Dimensions,Dimensions);
		lambda.clear();
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i,i) = Rng::uni(1,3.0);
		}
		RealMatrix A = createRandomMatrix(lambda,Dimensions);
		//calculate Cholesky
		RealMatrix C(Dimensions,Dimensions);
		choleskyDecomposition(A,C);
		
		//generate proper update
		double sigma = Rng::uni(-0.8,2);
		RealVector v(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			v(i) = Rng::uni(-1,1);
		}
		if(sigma < 0)//preserve positive definiteness
			v /= norm_2(v);
		
		//update decomposition
		noalias(A) += sigma*outer_prod(v,v);
		RealMatrix CUpdate=C;
		choleskyDecomposition(A,CUpdate);
		
		//Test the fast update
		choleskyUpdate(C,v,sigma);
		
		BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(C)));//test for nans
		BOOST_CHECK_SMALL(max(abs(C-CUpdate)),1.e-12);
	}
}
