#define BOOST_TEST_MODULE BLAS_Matrix_Expression
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>

using namespace shark;
using namespace blas;

std::size_t Dimensions = 10;

BOOST_AUTO_TEST_SUITE (BLAS_matrix_expression)


/////////////////////////////////////////////////////
///////////Matrix Reductions   ///////////
/////////////////////////////////////////////////////


BOOST_AUTO_TEST_CASE( BLAS_Matrix_Max )
{
	matrix<double> A(Dimensions, Dimensions,0.0); 
	double result = Dimensions*Dimensions-1;
	
	for (size_t i = 0; i < Dimensions; i++){
		for (size_t j = 0; j < Dimensions; j++){
			A(i,j) = i*Dimensions+j;
		}
	}
	matrix<double,column_major> Atrans = trans(A);
	BOOST_CHECK_CLOSE(max(A),result,1.e-10);
	BOOST_CHECK_CLOSE(max(Atrans),result,1.e-10);
}
BOOST_AUTO_TEST_CASE( BLAS_Matrix_Min )
{
	matrix<double> A(Dimensions, Dimensions,0.0); 
	double result = 0;
	
	for (size_t i = 0; i < Dimensions; i++){
		for (size_t j = 0; j < Dimensions; j++){
			A(i,j) = i*Dimensions+j;
		}
	}
	matrix<double,column_major> Atrans = trans(A);
	BOOST_CHECK_CLOSE(min(A),result,1.e-10);
	BOOST_CHECK_CLOSE(min(Atrans),result,1.e-10);
}

BOOST_AUTO_TEST_CASE( BLAS_Matrix_Sum )
{
	matrix<double> A(Dimensions, Dimensions,0.0); 
	double result = 0;
	
	for (size_t i = 0; i < Dimensions; i++){
		for (size_t j = 0; j < Dimensions; j++){
			A(i,j) = i*Dimensions+j;
			result += A(i,j);
		}
	}
	matrix<double,column_major> Atrans = trans(A);
	BOOST_CHECK_CLOSE(sum(A),result,1.e-10);
	BOOST_CHECK_CLOSE(sum(Atrans),result,1.e-10);
}

BOOST_AUTO_TEST_CASE( BLAS_Matrix_Norm_1_inf )
{
	matrix<double> A(Dimensions, Dimensions,0.0); 
	vector<double> colSum(Dimensions,0.0);
	vector<double> rowSum(Dimensions,0.0);
	
	for (size_t i = 0; i < Dimensions; i++){
		for (size_t j = 0; j < Dimensions; j++){
			A(i,j) = i*Dimensions+j-12.0;
			colSum(j) += std::abs(A(i,j));
			rowSum(i) += std::abs(A(i,j));
		}
	}
	double result1 = max(colSum);
	double resultInf = max(rowSum);
	matrix<double,column_major> Atrans = trans(A);
	BOOST_CHECK_CLOSE(norm_1(A),result1,1.e-10);
	BOOST_CHECK_CLOSE(norm_inf(A),resultInf,1.e-10);
	BOOST_CHECK_CLOSE(norm_1(Atrans),resultInf,1.e-10);
	BOOST_CHECK_CLOSE(norm_inf(Atrans),result1,1.e-10);
}

BOOST_AUTO_TEST_SUITE_END()
