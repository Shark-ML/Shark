#define BOOST_TEST_MODULE LinAlg_Repeat
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <shark/Core/Timer.h>

#include <shark/LinAlg/Base.h>

using namespace shark;

BOOST_AUTO_TEST_CASE( LinAlg_Repeat_Indexed ){
	IntVector vector(3);
	vector(0) = 1;
	vector(1) = 2;
	vector(2) = 3;
	
	blas::VectorRepeater<IntVector> rep=blas::repeat(vector,3);
	
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			BOOST_CHECK_EQUAL(rep(i,j), vector(j)); 
		}
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_Repeat_Iterator ){
	IntVector vector(3);
	vector(0) = 1;
	vector(1) = 2;
	vector(2) = 3;
	
	blas::VectorRepeater<IntVector> rep=blas::repeat(vector,3);
	 
	//test both iterator orders
	for(blas::VectorRepeater<IntVector>::const_iterator1 i = rep.begin1(); i != rep.end1(); ++i){
		std::size_t k = 0;
		for(blas::VectorRepeater<IntVector>::const_iterator2 j = i.begin(); j != i.end(); ++j,++k){
			BOOST_CHECK_EQUAL(*j, vector(k)); 
		}
	}
	
	std::size_t k = 0;
	for(blas::VectorRepeater<IntVector>::const_iterator2 i = rep.begin2(); i != rep.end2(); ++i,++k){
		for(blas::VectorRepeater<IntVector>::const_iterator1 j = i.begin(); j != i.end(); ++j){
			BOOST_CHECK_EQUAL(*j, vector(k)); 
		}
	}
}

//some expressions to test whether everything works together with ublas
BOOST_AUTO_TEST_CASE( LinAlg_Repeat_Expressions ){
	IntVector vector(3);
	vector(0) = 1;
	vector(1) = 2;
	vector(2) = 3;
	
	blas::VectorRepeater<IntVector> rep=blas::repeat(vector,3);
	
	RealMatrix result(rep);
	result+=blas::repeat(vector,3);
	
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 3; ++j){
			BOOST_CHECK_EQUAL(result(i,j), 2*vector(j)); 
		}
	}
}
