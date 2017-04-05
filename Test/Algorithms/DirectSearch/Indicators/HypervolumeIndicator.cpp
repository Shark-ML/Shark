#define BOOST_TEST_MODULE DirectSearch_HypervolumeIndicator
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Indicators/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Core/Random.h>
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
				population[i][j]= random::uni(random::globalRng,-1,10);
			}
		}
		//archive is empty
		std::vector<RealVector> archive;
		
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
		std::size_t indicated = indicator.leastContributor(population, archive);
		BOOST_CHECK_CLOSE(maxVolume,volumes[indicated],1.e-10);
	}
}


BOOST_AUTO_TEST_CASE( HypervolumeIndicator_LeastK ) {
	std::size_t numPoints = 10;
	std::size_t numTrials = 50;
	std::size_t numDims = 3;
	std::size_t K = 5;
	for(std::size_t t = 0; t != numTrials; ++t){ 
		//create points
		std::vector<RealVector> population(numPoints);
		std::vector<std::size_t> indices(numPoints);
		for(std::size_t i = 0; i != numPoints; ++i){
			population[i].resize(numDims);
			for(std::size_t j = 0; j != numDims; ++j){
				population[i][j]= random::uni(random::globalRng,-1,10);
			}
			indices[i] = i;
		}
		std::vector<RealVector> populationTest = population;
		std::set<std::size_t> indicesTest(indices.begin(),indices.end());
		
		//archive is empty
		std::vector<RealVector> archive;
		RealVector ref(numDims,11);
		HypervolumeIndicator indicator;
		indicator.setReference(ref);
		//remove K points one after another
		for(std::size_t k = 0; k != K; ++k){
			std::size_t indicated = indicator.leastContributor(population, archive);
			population.erase(population.begin() + indicated);
			indices.erase(indices.begin() + indicated);
		}
		//remove all K at the same time
		std::vector<std::size_t> indicated = indicator.leastContributors(populationTest, archive,K);
		for(auto pos:indicated){
			indicesTest.erase(pos);
		}
		//check that both sets are the same
		BOOST_REQUIRE_EQUAL(indicated.size(), K);
		BOOST_REQUIRE_EQUAL(indicesTest.size(), indices.size());
		
		auto pos = indicesTest.begin();
		for(std::size_t i = 0; i != indicesTest.size(); ++i,++pos){
			BOOST_CHECK_EQUAL(*pos, indices[i]);
		}
		
		
	}
}
BOOST_AUTO_TEST_SUITE_END()
