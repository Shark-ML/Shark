#define BOOST_TEST_MODULE LinAlg_Permute
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/Base.h>
using namespace shark;
BOOST_AUTO_TEST_CASE( LinAlg_Permute_Rows_Matrix ){
	PermutationMatrix P(5);
	P(0)=2;
	P(1)=2;
	P(2)=4;
	P(3)=4;
	P(4)=4;
	
	//generate testing matrix
	IntMatrix A(5,6);
	for(std::size_t i = 0; i != 5; ++i){
		for(std::size_t j = 0; j != 6; ++j){
			A(i,j) = i*6+j;
		}
	}
	//generate permutated result
	IntMatrix APerm(5,6);
	row(APerm,0)=row(A,2);
	row(APerm,1)=row(A,0);
	row(APerm,2)=row(A,4);
	row(APerm,3)=row(A,1);
	row(APerm,4)=row(A,3);
	
	
	swap_rows(P,A);
	int error = norm_inf(A-APerm);
	BOOST_CHECK_EQUAL(error, 0);
}

BOOST_AUTO_TEST_CASE( LinAlg_Permute_Rows_Vector ){
	PermutationMatrix P(5);
	P(0)=2;
	P(1)=2;
	P(2)=4;
	P(3)=4;
	P(4)=4;
	
	//generate testing matrix
	IntVector v(5);
	for(std::size_t i = 0; i != 5; ++i){
		v(i) = i;
	}
	//generate permutated result
	IntVector vPerm(5);
	vPerm(0)=v(2);
	vPerm(1)=v(0);
	vPerm(2)=v(4);
	vPerm(3)=v(1);
	vPerm(4)=v(3);
	
	swap_rows(P,v);
	
	int error = norm_inf(v-vPerm);
	BOOST_CHECK_EQUAL(error, 0);
}

BOOST_AUTO_TEST_CASE( LinAlg_Permute_Columns ){
	PermutationMatrix P(5);
	P(0)=2;
	P(1)=2;
	P(2)=4;
	P(3)=4;
	P(4)=4;
	
	//generate testing matrix
	IntMatrix A(6,5);
	for(std::size_t i = 0; i != 6; ++i){
		for(std::size_t j = 0; j != 5; ++j){
			A(i,j) = i*5+j;
		}
	}
	//generate permutated result
	IntMatrix APerm(6,5);
	column(APerm,0)=column(A,2);
	column(APerm,1)=column(A,0);
	column(APerm,2)=column(A,4);
	column(APerm,3)=column(A,1);
	column(APerm,4)=column(A,3);
	
	swap_columns(P,A);
	
	int error = norm_inf(A-APerm);
	BOOST_CHECK_EQUAL(error, 0);
}
BOOST_AUTO_TEST_CASE( LinAlg_Permute_Columns_Inverted ){
	PermutationMatrix P(5);
	P(0)=2;
	P(1)=2;
	P(2)=4;
	P(3)=4;
	P(4)=4;
	
	//generate testing matrix
	IntMatrix A(6,5);
	for(std::size_t i = 0; i != 6; ++i){
		for(std::size_t j = 0; j != 5; ++j){
			A(i,j) = i*5+j;
		}
	}
	//copy to result
	IntMatrix APerm(A);
	
	swap_columns(P,A);
	swap_columns_inverted(P,A);
	
	int error = norm_inf(A-APerm);
	BOOST_CHECK_EQUAL(error, 0);
}
BOOST_AUTO_TEST_CASE( LinAlg_Permute_Full ){
	PermutationMatrix P(5);
	P(0)=2;
	P(1)=2;
	P(2)=4;
	P(3)=4;
	P(4)=4;
	
	//generate testing matrix
	IntMatrix A(6,5);
	for(std::size_t i = 0; i != 6; ++i){
		for(std::size_t j = 0; j != 5; ++j){
			A(i,j) = i*5+j;
		}
	}
	//generate permutated result
	IntMatrix APerm(A);
	swap_rows(APerm,0,2);
	swap_columns(APerm,0,2);
	swap_rows(APerm,1,2);
	swap_columns(APerm,1,2);
	swap_rows(APerm,2,4);
	swap_columns(APerm,2,4);
	swap_rows(APerm,3,4);
	swap_columns(APerm,3,4);
	
	swap_full(P,A);
	
	int error = norm_inf(A-APerm);
	BOOST_CHECK_EQUAL(error, 0);
}
