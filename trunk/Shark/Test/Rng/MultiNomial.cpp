
#include <shark/Statistics/Distributions/MultiNomialDistribution.h>
#include <shark/Data/Statistics.h>

#define BOOST_TEST_MODULE Rng_MultiNomialDistribution
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

BOOST_AUTO_TEST_CASE( MultiNomial_same_probabilities) {
	std::size_t Samples = 100000;
	
	std::size_t Dimensions = 10;
	//Generate probability vector
	RealVector probabilities(Dimensions,1.0);
	
	MultiNomialDistribution dist(probabilities);
	RealVector draws(Dimensions,0.0);
	RealVector drawsDiscrete(Dimensions,0.0);
	for(std::size_t s = 0; s != Samples; ++s){
		MultiNomialDistribution::result_type sample = dist();
		BOOST_REQUIRE(sample < Dimensions);
		draws[sample]++;
		drawsDiscrete[Rng::discrete(0,Dimensions-1)]++;
	}
	draws/=Samples;
	drawsDiscrete/=Samples;
	
	double interval=(max(drawsDiscrete)-min(drawsDiscrete))/2;
	std::cout<<drawsDiscrete<<"\n"<<draws<<std::endl;
	std::cout<<"interval: "<<interval<<std::endl;
	for(std::size_t i = 0; i != Dimensions; ++i){
		BOOST_CHECK(draws(i) > 1.0/Dimensions -interval*1.5);
		BOOST_CHECK(draws(i) < 1.0/Dimensions + interval*1.5);
	}
}

BOOST_AUTO_TEST_CASE( MultiNomial_different_probabilities) {
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
