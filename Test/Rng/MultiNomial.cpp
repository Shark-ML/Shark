
#include <shark/Statistics/Distributions/MultiNomialDistribution.h>
#include <shark/Data/Statistics.h>

#define BOOST_TEST_MODULE Rng_MultiNomialDistribution
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

BOOST_AUTO_TEST_CASE( MultiNomialL_Cholesky) {
	std::size_t Samples = 1000000;
	std::size_t trials = 10;
	
	for(std::size_t t = 0; t != trials; ++t){
		std::size_t Dimensions = Rng::discrete(6,10);
		//Generate probability vector
		RealVector probabilities(Dimensions);
		
		for(std::size_t i = 0; i != Dimensions; ++i){
			probabilities(i) = Rng::uni(0.5,2);
		}
		probabilities[5] = 0;//make it a bit harder by creating a state which is not going to be drawn
		RealVector probabilitiesNormalized = probabilities/sum(probabilities);
		
		MultiNomialDistribution dist(probabilities);
		RealVector draws(Dimensions,0.0);
		for(std::size_t s = 0; s != Samples; ++s){
			MultiNomialDistribution::result_type sample = dist();
			BOOST_REQUIRE(sample < Dimensions);
			draws[sample]++;
		}
		draws/=Samples;
		
		for(std::size_t i = 0; i != Dimensions; ++i){
			BOOST_CHECK_CLOSE(draws(i),probabilitiesNormalized(i),1);
		}
	}
}
