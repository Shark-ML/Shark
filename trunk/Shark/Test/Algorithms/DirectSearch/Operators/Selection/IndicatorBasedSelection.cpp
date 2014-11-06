#define BOOST_TEST_MODULE DirectSearch_IndicatorBasedSelection
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/Indicators/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Rng/GlobalRng.h>

using namespace shark;

//generate random populations and check that the selection chooses the right ones.
BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_Operators_Selection_IndicatorBasedSelection)

BOOST_AUTO_TEST_CASE( IndicatorBasedSelection_Test ) {
	std::size_t numPoints = 50;
	std::size_t numTrials = 100;
	std::size_t numDims = 3;
	typedef IndicatorBasedSelection<HypervolumeIndicator> Selection;
	for(std::size_t t = 0; t != numTrials; ++t){ 
		//create points
		std::vector<Individual<RealVector,RealVector> > population(numPoints);
		for(std::size_t i = 0; i != numPoints; ++i){
			population[i].penalizedFitness().resize(numDims);
			for(std::size_t j = 0; j != numDims; ++j){
				population[i].penalizedFitness()[j]= Rng::uni(-1,2);
			}
		}
		
		std::size_t mu = Rng::discrete(1,numPoints-2);
		
		//store copy and compute ranks
		std::vector<Individual<RealVector,RealVector> > popCopy = population;
		FastNonDominatedSort sorter;
		sorter(popCopy);
		
		//store ranks in container and sort
		std::vector<unsigned int> ranks(numPoints);
		for(std::size_t i = 0; i != numPoints; ++i){
			ranks[i] = popCopy[i].rank();
		}
		std::sort(ranks.begin(),ranks.end());
		//no selected individual can have higher rank.
		std::size_t maxRank = ranks[mu];
		
		//do the selection 
		Selection selection;
		selection(population,mu);
		
		//check1: ranks are the same!
		for(std::size_t i = 0; i != numPoints; ++i){
			BOOST_CHECK_EQUAL(population[i].rank(), popCopy[i].rank());
		}
		
		//check2: we selected the right amount
		//check3: no selected individual has higher rank than maxrank
		std::size_t numSelected = 0;
		for(std::size_t i = 0; i != numPoints; ++i){
			if(population[i].selected()){
				++numSelected;
				BOOST_CHECK(population[i].rank() <= maxRank);
			}
		}
		BOOST_CHECK_EQUAL(numSelected,mu);
		
		//~ //check that ranks are okay
		//~ for(std::size_t i = 0; i != numPoints; ++i){
			//~ for(std::size_t j = 0; j != numPoints; ++j){
				//~ int comp = pdc(population[i],population[j]);
				//~ if(comp > 1){//i dominates j
					//~ BOOST_CHECK(population[i].rank() < population[j].rank() );
				//~ } else if (comp < -1){//j dominates i
					//~ BOOST_CHECK(population[i].rank() > population[j].rank() );
				//~ }
				
				//~ if(population[i].rank() == population[j].rank()){
					//~ BOOST_CHECK(comp == 1 || comp == -1);
				//~ }
			//~ }
		//~ }
	}
}
BOOST_AUTO_TEST_SUITE_END()
