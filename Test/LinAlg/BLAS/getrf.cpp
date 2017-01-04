#define BOOST_TEST_MODULE BLAS_Getrf
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Core/Shark.h>
#include <shark/LinAlg/BLAS/kernels/getrf.hpp>
#include <shark/LinAlg/BLAS/matrix.hpp>
#include <shark/LinAlg/BLAS/matrix_expression.hpp>
#include <shark/LinAlg/BLAS/matrix_proxy.hpp>
#include <shark/LinAlg/BLAS/io.hpp>

using namespace shark;

//the matrix is designed such that permutation will always give the next row
blas::matrix<double> createMatrix(std::size_t dimensions){
	blas::matrix<double> L(dimensions,dimensions,0.0);
	blas::matrix<double> U(dimensions,dimensions,0.0);
	
	for(std::size_t i = 0; i != dimensions; ++i){
		for(std::size_t j = 0; j <i; ++j){
			U(j,i) = 2.0/std::abs((int)i -(int)j);
			L(i,j)  = 5 - 5.0/dimensions*std::abs((int)i -(int)j);
		}
		U(i,i) = 0.5/dimensions*i+1;
		L(i,i) = 1;
	}
	blas::matrix<double> A = prod(L,U);
	return A;
}
typedef boost::mpl::list<blas::row_major,blas::column_major> result_orientations;


BOOST_AUTO_TEST_SUITE (BLAS_Cholesky)

BOOST_AUTO_TEST_CASE_TEMPLATE(BLAS_Potrf, Orientation,result_orientations) {
	std::size_t Dimensions = 123;
	//first generate a suitable eigenvalue problem matrix A
	blas::matrix<double,Orientation> A = createMatrix(Dimensions);
	//calculate lu decomposition
	blas::permutation_matrix P(Dimensions);
	blas::matrix<double,Orientation> dec = A;
	blas::kernels::getrf(dec,P);

	//copy upper matrix to temporary
	blas::matrix<double> upper(Dimensions,Dimensions,0.0);
	for (size_t row = 0; row < Dimensions; row++){
		for (size_t col = row; col < Dimensions ; col++){
			upper(row, col) = dec(row, col);
		}
	}
	
	//create reconstruction of A
	blas::matrix<double> testA = blas::triangular_prod<blas::unit_lower>(dec,upper);
	swap_rows_inverted(P,testA);
	
	//test reconstruction error
	double error = max(abs(A - testA));
	BOOST_CHECK_SMALL(error,1.e-12);
	BOOST_CHECK(!(boost::math::isnan)(norm_frobenius(testA)));//test for nans
}

BOOST_AUTO_TEST_SUITE_END()
