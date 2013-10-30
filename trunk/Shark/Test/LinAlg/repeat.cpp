#define BOOST_TEST_MODULE LinAlg_Repeat
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <shark/Core/Timer.h>

#include <shark/LinAlg/Base.h>

using namespace shark;

BOOST_AUTO_TEST_CASE( LinAlg_Repeat_Indexed ){
	std::size_t dimensions = 3;
	std::size_t repetitions = 10;
	IntVector vector(dimensions);
	vector(0) = 1;
	vector(1) = 2;
	vector(2) = 3;
	
	blas::vector_repeater<IntVector> rep=repeat(vector,repetitions);
	BOOST_REQUIRE_EQUAL(rep.size1(), repetitions);
	BOOST_REQUIRE_EQUAL(rep.size2(), dimensions);
	
	for(std::size_t i = 0; i != repetitions; ++i){
		for(std::size_t j = 0; j != dimensions; ++j){
			BOOST_CHECK_EQUAL(rep(i,j), vector(j)); 
		}
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_Repeat_Iterator ){
	std::size_t dimensions = 3;
	std::size_t repetitions = 10;
	IntVector vector(dimensions);
	vector(0) = 1;
	vector(1) = 2;
	vector(2) = 3;
	
	blas::vector_repeater<IntVector> rep=repeat(vector,repetitions);
	BOOST_REQUIRE_EQUAL(rep.size1(), repetitions);
	BOOST_REQUIRE_EQUAL(rep.size2(), dimensions);
	 
	//test both iterator orders
	for(std::size_t i = 0; i != repetitions; ++i){
		std::size_t k = 0;
		for(blas::vector_repeater<IntVector>::const_row_iterator j = rep.row_begin(i); j != rep.row_end(i); ++j,++k){
			BOOST_CHECK_EQUAL(*j, vector(k)); 
		}
		BOOST_REQUIRE_EQUAL(k, dimensions);
	}
	
	for(std::size_t i = 0; i != dimensions; ++i){
		std::size_t k = 0;
		for(blas::vector_repeater<IntVector>::const_column_iterator j = rep.column_begin(i); 
			j != rep.column_end(i); ++j,++k
		){
			BOOST_CHECK_EQUAL(*j, vector(i)); 
		}
		BOOST_REQUIRE_EQUAL(k, repetitions);
	}
}

//some expressions to test whether everything works together with ublas
BOOST_AUTO_TEST_CASE( LinAlg_Repeat_Expressions ){
	std::size_t dimensions = 3;
	std::size_t repetitions = 10;
	IntVector vector(dimensions);
	vector(0) = 1;
	vector(1) = 2;
	vector(2) = 3;
	
	blas::vector_repeater<IntVector> rep=repeat(vector,repetitions);
	BOOST_REQUIRE_EQUAL(rep.size1(), repetitions);
	BOOST_REQUIRE_EQUAL(rep.size2(), dimensions);
	
	RealMatrix result(rep);
	result+=blas::repeat(vector,repetitions);
	
	for(std::size_t i = 0; i != repetitions; ++i){
		for(std::size_t j = 0; j != dimensions; ++j){
			BOOST_CHECK_EQUAL(result(i,j), 2*vector(j)); 
		}
	}
}
