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
	double inputs[]={0.5,1,1.5,2.0};
	double values[]={sqr(pi)/2.0,sqr(pi)/6,sqr(pi)/2-4,sqr(pi)/6-1};
	
	double maxError = 1.e-12;
	for(std::size_t i = 0; i != 4; ++i){
		double x = Rng::uni(0,5);
		double y=trigamma(inputs[i]);
		
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

