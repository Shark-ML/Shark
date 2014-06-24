#include "shark/LinAlg/eigenvalues.h"
#include "shark/LinAlg/rotations.h"

#define BOOST_TEST_MODULE LinAlg_eigensymm
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;
using namespace blas;

BOOST_AUTO_TEST_CASE( LinAlg_eigensymm )
{
	RealMatrix A(3, 3);   // input matrix
	RealMatrix x(3, 3);   // matrix for eigenvectors
	RealVector lambda(3); // vector for eigenvalues

	// Initialization values for input matrix:
	double bottom_triangle[9] =
		{
			7.,  0.,  0.,

			-2.,  6.,  0.,

			0., -2.,  0.
		};

	// Initializing matrices and vector:
	for (size_t curr_row = 0; curr_row < 3; curr_row++)
	{
		for (size_t curr_col = 0; curr_col < 3; curr_col++)
		{
			A(curr_row, curr_col) = bottom_triangle[curr_row*3+curr_col];
			x(curr_row, curr_col) = 0.;
		}
		lambda(curr_row) = 0.;
	}

	// Calculating eigenvalues and eigenvectors:
	eigensymm(A, x, lambda);// A is unchanged after the call

	// lines below are for self-testing this example, please ignore
	BOOST_CHECK_SMALL(lambda(0)- 8.74697,1.e-5);
}


RealMatrix createRandomMatrix(RealVector const& lambda,std::size_t Dimensions){
	RealMatrix R = blas::randomRotationMatrix(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		column(R,i) *= std::sqrt(lambda(i));
	}
	RealMatrix A(Dimensions,Dimensions);
	axpy_prod(R,trans(R),A);
	return A;
}

BOOST_AUTO_TEST_CASE( LinAlg_eigensymm_general )
{
	std::size_t NumTests = 100;
	std::size_t Dimensions = 10;
	for(std::size_t test = 0; test != NumTests; ++test){
		//first generate a suitable eigenvalue problem matrix A as well as its decompisition
		RealVector lambda(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			lambda(i) = Rng::uni(1,3.0);
		}
		RealMatrix A = createRandomMatrix(lambda,Dimensions);
		std::sort(lambda.begin(),lambda.end());
		std::reverse(lambda.begin(),lambda.end());
		
		//calculate eigenvalue decomposition
		RealVector eigenvalues;
		RealMatrix eigenVectors;
		eigensymm(A, eigenVectors, eigenvalues);
		
		//check that the eigenvectors are orthogonal
		double error1 = max(prod(eigenVectors,trans(eigenVectors))-identity_matrix<double>(Dimensions));
		double error2 = max(prod(trans(eigenVectors),eigenVectors)-identity_matrix<double>(Dimensions));
		
		BOOST_CHECK(!(boost::math::isnan)(error1));//test for nans
		BOOST_CHECK(!(boost::math::isnan)(error2));//test for nans
		BOOST_CHECK_SMALL(error1,1.e-12);
		BOOST_CHECK_SMALL(error2,1.e-12);
		
		//check that the eigenvalues are correct
		//~ std::cout<<eigenvalues<<std::endl;
		//~ std::cout<<lambda<<std::endl;
		BOOST_CHECK_SMALL(norm_inf(eigenvalues - lambda),1.e-12);
		
		//check that the eigenvectors actually orthogonalize the matrix
		double error_orthogonalize = max(
			prod(prod(trans(eigenVectors),A),eigenVectors)-diagonal_matrix<RealVector>(eigenvalues)
		);
		BOOST_CHECK(!(boost::math::isnan)(error_orthogonalize));//test for nans
		BOOST_CHECK_SMALL(error_orthogonalize,1.e-12);
	}
}