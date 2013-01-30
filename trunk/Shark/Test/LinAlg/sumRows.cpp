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
	RealVector test1(columns);
	RealVector test2(columns);
	RealVector testResult(columns);
	
	for(std::size_t j = 0; j != columns; ++j){
		for(std::size_t i = 0; i != rows; ++i){
			A(i,j) = Rng::uni(0,1);
		} 
		test1(j) = test2(j) = testResult(j) = Rng::uni(0,1);
	}
	RealMatrix Atrans=trans(A);

	//test implementation
	for(std::size_t i = 0; i != rows; ++i){
		testResult+=row(A,i);
	}
	
	//sumRows with row major argument
	sumRows(A,test1);
	double error = norm_2(testResult-test1);
	BOOST_CHECK_SMALL(error,1.e-15);
	
	//sumRows with column major argument
	sumRows(trans(Atrans),test2);
	error = norm_2(testResult-test2);
	BOOST_CHECK_SMALL(error,1.e-15);
}
BOOST_AUTO_TEST_CASE( LinAlg_sumColumns){
	std::size_t rows = 101;
	std::size_t columns = 89;
	RealMatrix A(rows, columns);
	RealVector test1(rows);
	RealVector test2(rows);
	RealVector testResult(rows);
	
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			A(i,j) = Rng::uni(0,1);
		} 
		test1(i) = test2(i) = testResult(i) = Rng::uni(0,1);
	}
	RealMatrix Atrans=trans(A);

	//test implementation
	for(std::size_t i = 0; i != columns; ++i){
		testResult+=column(A,i);
	}
	
	//sumRows with row major argument
	sumColumns(A,test1);
	double error = norm_2(testResult-test1);
	BOOST_CHECK_SMALL(error,1.e-15);
	
	//sumRows with column major argument
	sumColumns(trans(Atrans),test2);
	error = norm_2(testResult-test2);
	BOOST_CHECK_SMALL(error,1.e-15);
}
#ifdef NDEBUG
BOOST_AUTO_TEST_CASE( LinAlg_sumRows_BENCHMARK){
	std::size_t rows = 28*27;
	std::size_t columns = 500;
	std::size_t iterations = 1000;
	RealMatrix A(rows, columns);
	RealVector testResult(columns);
	
	for(std::size_t j = 0; j != columns; ++j){
		for(std::size_t i = 0; i != rows; ++i){
			A(i,j) = Rng::uni(0,1);
		} 
		testResult(j) = Rng::uni(0,1);
	}
	RealMatrix Atrans=trans(A);
	
	std::cout<<"Benchmarking sumRows"<<std::endl;
	
	double start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		for(std::size_t j = 0; j != rows; ++j){
			noalias(testResult)+=row(A,j);
		}
	}
	double end=Timer::now();
	testResult/=10000;
	std::cout<<"naiive solution row major: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			testResult(j)+=sum(row(Atrans,j));
		}
	}
	end=Timer::now();
	std::cout<<"naiive solution column major: "<<end-start<<std::endl;
	testResult/=10000;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		sumRows(A,testResult);
	}
	end=Timer::now();
	testResult/=10000;
	std::cout<<"sumRows row major: "<<end-start<<std::endl;
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		sumRows(trans(Atrans),testResult);
	}
	end=Timer::now();
	testResult/=10000;
	std::cout<<"sumRows column major: "<<end-start<<std::endl;
	
	
	double output= 0;
	output += inner_prod(testResult,testResult);
	std::cout<<"anti optimization output: "<<output<<std::endl;
	
}

BOOST_AUTO_TEST_CASE( LinAlg_sumMatrix_BENCHMARK){
	std::size_t rows = 28*27;
	std::size_t columns = 500;
	std::size_t iterations = 100;
	RealMatrix A(rows, columns);
	double testResult = 0;
	
	for(std::size_t j = 0; j != columns; ++j){
		for(std::size_t i = 0; i != rows; ++i){
			A(i,j) = Rng::uni(0,1);
		} 
	}
	RealMatrix Atrans=trans(A);
	
	std::cout<<"Benchmarking sumComplete"<<std::endl;
	
	double start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		for(std::size_t j = 0; j != rows; ++j){
			testResult+=sum(row(sqr(A),j));
		}
	}
	double end=Timer::now();
	testResult/=10000;
	std::cout<<"naiive solution row major: "<<end-start<<std::endl;
	
	start=Timer::now();
	for(std::size_t i = 0; i != iterations; ++i){
		testResult+=sum(sumRows(sqr(A)));
	}
	end=Timer::now();
	testResult/=10000;
	std::cout<<"sumRows row major: "<<end-start<<std::endl;
	
	std::cout<<"anti optimization output: "<<testResult<<std::endl;
	
}

#endif
