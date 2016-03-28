#define BOOST_TEST_MODULE DirectSearch_FastNonDominatedSort
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/FastNonDominatedSort.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Rng/GlobalRng.h>

using namespace shark;

//randomly creates populations of individuals, sorts them and checks that the ranks are okay.
BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_FastNonDominatedSort)

BOOST_AUTO_TEST_CASE( FastNonDominatedSort_Test ) {
	std::size_t numPoints = 50;
	std::size_t numTrials = 10;
	std::size_t numDims = 3;
	for(std::size_t t = 0; t != numTrials; ++t){ 
		//create points
		std::vector<Individual<RealVector,RealVector> > population(numPoints);
		for(std::size_t i = 0; i != numPoints; ++i){
			population[i].penalizedFitness().resize(numDims);
			for(std::size_t j = 0; j != numDims; ++j){
				population[i].penalizedFitness()[j]= Rng::uni(-1,2);
			}
		}

		FastNonDominatedSort sorter;
		sorter(population);

		//check that ranks are okay
		for(std::size_t i = 0; i != numPoints; ++i){
			for(std::size_t j = 0; j != numPoints; ++j){
				ParetoRelation rel = dominance<Individual<RealVector,RealVector>, FitnessExtractor>(population[i],population[j]);
				if(prec(rel)){//i dominates j
					BOOST_CHECK(population[i].rank() < population[j].rank() );
				} else if (succ(rel)){//j dominates i
					BOOST_CHECK(population[i].rank() > population[j].rank() );
				}

				if(population[i].rank() == population[j].rank()){
					BOOST_CHECK(rel == INCOMPARABLE || rel == EQUIVALENT);
				}
			}
		}
	}
}
BOOST_AUTO_TEST_SUITE_END()
