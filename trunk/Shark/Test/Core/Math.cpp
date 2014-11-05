#define BOOST_TEST_MODULE Core_Math
#include <shark/Core/Math.h>
#include <shark/Rng/GlobalRng.h>
#include <boost/math/special_functions/digamma.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
using namespace shark;

BOOST_AUTO_TEST_CASE(Math_Trigamma_SpecialValues)
{
	double pi = boost::math::constants::pi<double>();
	double inputs[]={0.25,0.5,1,1.5,2.0};
	double K = 0.915965594177219015054;
	double values[]={sqr(pi)+8*K,sqr(pi)/2.0,sqr(pi)/6,sqr(pi)/2-4,sqr(pi)/6-1};
	
	double maxError = 1.e-12;
	for(std::size_t i = 0; i != 4; ++i){
		double y=trigamma(inputs[i]);
		
		std::cout<<inputs[i]<<" "<<values[i]<<" "<<y<<std::endl;
		BOOST_CHECK_CLOSE(y,values[i],maxError);
	}
}
BOOST_AUTO_TEST_CASE(Math_Tetragamma_SpecialValues)
{
	double pi = boost::math::constants::pi<double>();
	double inputs[]={1.0/2000,0.25,0.5,1,1.5,2.0,10};
	//values computedusing wolfram alpha - they should get it right
	double values[]={
		-1.6000000002400869e10,//tests taylor expansion for small values
		-129.3277399375369203,
		-16.82879664423431999,
		-2.404113806319188570,
		-0.828796644234319995,
		-0.40411380631918857079,
		-0.0110498349708020674621
	};
	double maxError = 1.e-12;
	for(std::size_t i = 0; i != 7; ++i){
		double y=tetragamma(inputs[i]);
		
		std::cout<<inputs[i]<<" "<<values[i]<<" "<<y<<std::endl;
		BOOST_CHECK_CLOSE(y,values[i],maxError);
	}
}
BOOST_AUTO_TEST_CASE(Math_Trigamma_Random)
{
	std::size_t numTrials = 1000;
	double epsilon = 1.e-8;
	double maxError = 1.e-6;
	for(std::size_t trial = 0; trial != numTrials; ++trial){
		double x = Rng::uni(0,5);
		double y=trigamma(x);
		double yEst= (boost::math::digamma((long double)(x+epsilon))-boost::math::digamma((long double)(x-epsilon)))/(2.0*epsilon);
		
		BOOST_CHECK_CLOSE(y,yEst,maxError);
	}
}

