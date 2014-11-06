#define BOOST_TEST_MODULE LinAlg_Sum_Rows
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Core/Timer.h>
#include <shark/LinAlg/Base.h>
#include <shark/Rng/GlobalRng.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (LinAlg_sumRows)

BOOST_AUTO_TEST_CASE( LinAlg_sum_rows){
	std::size_t rows = 101;
	std::size_t columns = 89;
	RealMatrix A(rows, columns);
	RealVector testResult(columns,0.0);
	
	for(std::size_t j = 0; j != columns; ++j){
		for(std::size_t i = 0; i != rows; ++i){
			A(i,j) = Rng::uni(0,1);
		}
	}
	RealMatrix Atrans = trans(A);

	//test implementation
	for(std::size_t i = 0; i != rows; ++i){
		testResult+=row(A,i);
	}
	
	//sum_rows with row major argument
	RealVector test1 = sum_rows(A);
	double error = norm_2(testResult-test1);
	BOOST_CHECK_SMALL(error,1.e-15);
	
	//sum_rows with column major argument
	RealVector test2 = sum_rows(trans(Atrans));
	error = norm_2(testResult-test2);
	BOOST_CHECK_SMALL(error,1.e-15);
}
BOOST_AUTO_TEST_CASE( LinAlg_sum_columns){
	std::size_t rows = 101;
	std::size_t columns = 89;
	RealMatrix A(rows, columns);
	RealVector testResult(rows,0.0);
	
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			A(i,j) = Rng::uni(0,1);
		} 
	}
	RealMatrix Atrans = trans(A);

	//test implementation
	for(std::size_t i = 0; i != columns; ++i){
		testResult+=column(A,i);
	}
	
	//sum_rows with row major argument
	RealVector test1 = sum_columns(A);
	double error = norm_2(testResult-test1);
	BOOST_CHECK_SMALL(error,1.e-15);
	
	//sum_rows with column major argument
	RealVector test2 = sum_columns(trans(Atrans));
	error = norm_2(testResult-test2);
	BOOST_CHECK_SMALL(error,1.e-15);
}

BOOST_AUTO_TEST_SUITE_END()
