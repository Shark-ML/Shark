#include <shark/LinAlg/rotations.h>
#include <shark/Rng/GlobalRng.h>
#define BOOST_TEST_MODULE LinAlg_rotations
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

BOOST_AUTO_TEST_CASE( LinAlg_Random_Rotation_Matrix ){
	std::size_t NumTests = 1;
	std::size_t Dimensions = 50;
	Rng::seed(42);
	RealMatrix result(Dimensions,Dimensions);
	for(std::size_t test = 0;test!=NumTests;++test){

		//test whether R^TR = RR^T = I
		RealMatrix R = blas::randomRotationMatrix(Dimensions);
		
		for(std::size_t i = 0; i != Dimensions; ++i){
			BOOST_CHECK_SMALL(norm_2(row(R,i))-1,1.e-12);
			BOOST_CHECK_SMALL(norm_2(column(R,i))-1,1.e-12);
		}
		
		result.clear();
		fast_prod(R,trans(R),result,0);
		double errorID1 = norm_inf(result-RealIdentityMatrix(Dimensions));
		result.clear();
		fast_prod(trans(R),R,result,0);
		double errorID2 = norm_inf(result-RealIdentityMatrix(Dimensions));
		
		BOOST_CHECK_SMALL(errorID1,1.e-13);
		BOOST_CHECK_SMALL(errorID2,1.e-13);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_Householder_Creation ){
	///test for numerical stability of createHouseholderReflection
	std::size_t NumTests = 1000;
	std::size_t Dimensions = 200;
	Rng::seed(42);//our all loved default seed :)
	for(std::size_t testi = 0;testi!=NumTests;++testi){
		RealVector test(Dimensions);
		RealVector reflection(Dimensions);
		for(std::size_t i = 0; i != Dimensions; ++i){
			test(i) = Rng::gauss(0,1);
		}
		double norm = -copySign(norm_2(test),test(0));
		

		double tau = createHouseholderReflection(test,reflection);
		
		//when the reflection is correct, it should lead to
		//(norm,0,....,0) when applied to test
		RealVector result = test;
		result -= tau*prod(outer_prod(reflection,reflection),test);
		
		
		BOOST_CHECK_SMALL(norm-result(0),1.e-13);
		for(size_t i=1; i != Dimensions; ++i){
			BOOST_CHECK_SMALL(result(i),1.e-13);
		}
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_Householder_Apply_Left ){
	///test for numerical stability of applyHouseholderOnTheLeft
	std::size_t NumTests = 100;
	std::size_t Dimension1 = 200;
	std::size_t Dimension2 = 10;
	Rng::seed(42);
	for(std::size_t testi = 0; testi != NumTests; ++testi){
		RealMatrix test(Dimension1,Dimension2);
		RealMatrix result(Dimension1,Dimension2);
		RealVector reflection(Dimension1);
		for(std::size_t i = 0; i != Dimension1; ++i){
			for(std::size_t j = 0; j != Dimension2; ++j){
				result(i,j) = test(i,j) = Rng::gauss(0,1);
			}
		}

		double tau = createHouseholderReflection(column(test,0),reflection);
		
		//apply the slow default version
		RealMatrix O = outer_prod(reflection,reflection);
		noalias(result) -= tau*prod(O,test);
		
		//now the real test
		applyHouseholderOnTheLeft(test,reflection,tau);
		
		//they should be similar
		for(std::size_t i = 0; i != Dimension1; ++i){
			for(std::size_t j = 0; j != Dimension2; ++j){
				BOOST_CHECK_SMALL(result(i,j)-test(i,j),1.e-13);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_Householder_Apply_Right ){
	///test for numerical stability of applyHouseholderOnTheRight
	std::size_t NumTests = 100;
	std::size_t Dimension1 = 200;
	std::size_t Dimension2 = 10;
	Rng::seed(42);
	for(std::size_t testi = 0; testi != NumTests; ++testi){
		RealMatrix test(Dimension1,Dimension2);
		RealMatrix result(Dimension1,Dimension2);
		RealVector reflection(Dimension2);
		for(std::size_t i = 0; i != Dimension1; ++i){
			for(std::size_t j = 0; j != Dimension2; ++j){
				result(i,j) = test(i,j) = Rng::gauss(0,1);
			}
		}

		double tau = createHouseholderReflection(row(test,0),reflection);
		
		//apply the slow default version
		RealMatrix O = outer_prod(reflection,reflection);
		noalias(result) -= tau*prod(test,O);
		
		//now the real test
		applyHouseholderOnTheRight(test,reflection,tau);
		
		//they should be similar
		for(std::size_t i = 0; i != Dimension1; ++i){
			for(std::size_t j = 0; j != Dimension2; ++j){
				BOOST_CHECK_SMALL(result(i,j)-test(i,j),1.e-13);
			}
		}
	}
}
