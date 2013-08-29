#define BOOST_TEST_MODULE LinAlg_Sum_Rows
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <shark/Core/Timer.h>
#include <shark/LinAlg/Base.h>
#include <shark/Rng/GlobalRng.h>

using namespace shark;

BOOST_AUTO_TEST_CASE( LinAlg_sumRows){
	std::size_t rows = 101;
	std::size_t columns = 89;
	RealMatrix A(rows, columns);
	RealVector testResult(columns,0.0);
	
	for(std::size_t j = 0; j != columns; ++j){
		for(std::size_t i = 0; i != rows; ++i){
			A(i,j) = Rng::uni(0,1);
		}
	}
	RealMatrix Atrans=trans(A);

	//test implementation
	for(std::size_t i = 0; i != rows; ++i){
		testResult+=row(A,i);
	}
	
	//sumRows with row major argument
	RealVector test1 = sumRows(A);
	double error = norm_2(testResult-test1);
	BOOST_CHECK_SMALL(error,1.e-15);
	
	//sumRows with column major argument
	RealVector test2 = sumRows(trans(Atrans));
	error = norm_2(testResult-test2);
	BOOST_CHECK_SMALL(error,1.e-15);
}
BOOST_AUTO_TEST_CASE( LinAlg_sumColumns){
	std::size_t rows = 101;
	std::size_t columns = 89;
	RealMatrix A(rows, columns);
	RealVector testResult(rows,0.0);
	
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			A(i,j) = Rng::uni(0,1);
		} 
	}
	RealMatrix Atrans=trans(A);

	//test implementation
	for(std::size_t i = 0; i != columns; ++i){
		testResult+=column(A,i);
	}
	
	//sumRows with row major argument
	RealVector test1 = sumColumns(A);
	double error = norm_2(testResult-test1);
	BOOST_CHECK_SMALL(error,1.e-15);
	
	//sumRows with column major argument
	RealVector test2 = sumColumns(trans(Atrans));
	error = norm_2(testResult-test2);
	BOOST_CHECK_SMALL(error,1.e-15);
}
