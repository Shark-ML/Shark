#define BOOST_TEST_MODULE DirectSearch_HypervolumeIndicator
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Indicators/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Rng/GlobalRng.h>
#include <limits>

using namespace shark;

//checks for random sets that the least contributor actually has the least contribution
BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_Indicators_HypervolumeIndicator)

BOOST_AUTO_TEST_CASE( HypervolumeIndicator_Consistency ) {
	std::size_t numPoints = 10;
	std::size_t numTrials = 50;
	std::size_t numDims = 2;
	for(std::size_t t = 0; t != numTrials; ++t){ 
		//create points
		std::vector<RealVector> population(numPoints);
		for(std::size_t i = 0; i != numPoints; ++i){
			population[i].resize(numDims);
			for(std::size_t j = 0; j != numDims; ++j){
				population[i][j]= Rng::uni(-1,10);
			}
		}
		
		
		RealVector ref(numDims,11);
		
		RealVector volumes(numPoints);
		double maxVolume = -std::numeric_limits<double>::max();
		for(std::size_t i = 0; i != numPoints; ++i){
			HypervolumeCalculator hv;
			std::vector<RealVector> copy = population;
			copy.erase(copy.begin()+i);
			
			
			double volume = hv(copy,ref);
			volumes[i] =volume;
			if(maxVolume < volume){
				maxVolume = volume;
			}
		}
		HypervolumeIndicator indicator;
		indicator.setReference(ref);
		std::size_t indicated = indicator.leastContributor(population);
		BOOST_CHECK_CLOSE(maxVolume,volumes[indicated],1.e-10);
	}
}
BOOST_AUTO_TEST_SUITE_END()
