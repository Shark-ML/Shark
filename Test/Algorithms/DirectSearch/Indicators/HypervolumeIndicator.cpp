#define BOOST_TEST_MODULE DirectSearch_HypervolumeIndicator
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Indicators/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Rng/GlobalRng.h>
#include <limits>

using namespace shark;

//checks for random sets that the least contributor actually has the least contribution
BOOST_AUTO_TEST_CASE( HypervolumeIndicator_Consistency ) {
	std::size_t numPoints = 10;
	std::size_t numTrials = 50;
	std::size_t numDims = 2;
	for(std::size_t t = 0; t != numTrials; ++t){ 
		//create points
		std::vector<Individual<RealVector,RealVector> > population(numPoints);
		for(std::size_t i = 0; i != numPoints; ++i){
			population[i].penalizedFitness().resize(numDims);
			for(std::size_t j = 0; j != numDims; ++j){
				population[i].penalizedFitness()[j]= Rng::uni(-1,10);
			}
		}
		
		
		
		RealVector ref(numDims,11);
		HypervolumeIndicator indicator;
		
		RealVector volumes(numPoints);
		
		double maxVolume = -std::numeric_limits<double>::max();
		for(std::size_t i = 0; i != numPoints; ++i){
			std::vector<Individual<RealVector,RealVector> > copy = population;
			copy.erase(copy.begin()+i);
			double volume = indicator(FitnessExtractor(),copy,ref);
			volumes[i] =volume;
			if(maxVolume < volume){
				maxVolume = volume;
			}
		}
		std::size_t indicated = indicator.leastContributor(FitnessExtractor(),population,ref);
		BOOST_CHECK_CLOSE(maxVolume,volumes[indicated],1.e-10);
	}
}