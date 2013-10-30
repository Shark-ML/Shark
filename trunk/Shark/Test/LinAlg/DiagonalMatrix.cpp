#define BOOST_TEST_MODULE LinAlg_Diagonal_Matrix
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/Base.h>

using namespace shark;

template<class M, class D>
void checkDiagonalMatrix(M const& diagonal, D const& diagonalElements, std::size_t const Dimensions){
	BOOST_REQUIRE_EQUAL(diagonal.size1(),Dimensions);
	BOOST_REQUIRE_EQUAL(diagonal.size2(),Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i){
		for(std::size_t j = 0; j != Dimensions; ++j){
			if(i != j)
				BOOST_CHECK_EQUAL(diagonal(i,j),0);
			else
				BOOST_CHECK_EQUAL(diagonal(i,i),diagonalElements(i));
		}
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_Diagonal_Matrix_Basic ){
	std::size_t const Dimensions = 10;
	IntVector diagonalElements(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i)
		diagonalElements(i) = i;
	
	blas::diagonal_matrix<IntVector> diagonal(diagonalElements);
	checkDiagonalMatrix(diagonal,diagonalElements,Dimensions);
}

BOOST_AUTO_TEST_CASE( LinAlg_Diagonal_Matrix_Copy ){
	std::size_t const Dimensions = 10;
	IntVector diagonalElements(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i)
		diagonalElements(i) = i;
	
	blas::diagonal_matrix<IntVector> diagonal(diagonalElements);
	
	blas::diagonal_matrix<IntVector> diagonal2(diagonal);
	checkDiagonalMatrix(diagonal2,diagonalElements,Dimensions);
}

BOOST_AUTO_TEST_CASE( LinAlg_Diagonal_Matrix_DefaultCtorAndAssignment){
	std::size_t const Dimensions = 10;
	IntVector diagonalElements(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i)
		diagonalElements(i) = i;
	
	blas::diagonal_matrix<IntVector> diagonal(diagonalElements);
	
	blas::diagonal_matrix<IntVector> diagonal2;
	BOOST_REQUIRE_EQUAL(diagonal2.size1(),0);
	BOOST_REQUIRE_EQUAL(diagonal2.size2(),0);
	
	//assignment
	diagonal2 = diagonal;
	checkDiagonalMatrix(diagonal2,diagonalElements,Dimensions);
}

BOOST_AUTO_TEST_CASE( LinAlg_Diagonal_Matrix_Assignment){
	std::size_t const Dimensions = 10;
	IntVector diagonalElements(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i)
		diagonalElements(i) = i;
	
	blas::diagonal_matrix<IntVector> diagonal(diagonalElements);
	
	IntMatrix diagonal2(diagonal);
	checkDiagonalMatrix(diagonal2,diagonalElements,Dimensions);
}

BOOST_AUTO_TEST_CASE( LinAlg_Identity_Matrix ){
	std::size_t const Dimensions = 10;
	IntVector diagonalElements(Dimensions);
	for(std::size_t i = 0; i != Dimensions; ++i)
		diagonalElements(i) = 1;
	
	blas::identity_matrix<int> diagonal(Dimensions);
	checkDiagonalMatrix(diagonal,diagonalElements,Dimensions);
}
