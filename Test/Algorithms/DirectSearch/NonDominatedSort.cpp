#define BOOST_TEST_MODULE DirectSearch_FastNonDominatedSort
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/FastNonDominatedSort.h>
#include <shark/Algorithms/DirectSearch/DCNonDominatedSort.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Core/Timer.h>

#include <cmath>
#include <cassert>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_NonDominatedSort)

// Create random populations of individuals, sorts them,
// checks that the ranks are okay, and check that the two
// tested algorithms deliver the same result.
BOOST_AUTO_TEST_CASE( NonDominatedSort_Test )
{
	std::size_t numPoints = 100;
	std::size_t numTrials = 10;
	for (std::size_t numDims=2; numDims <= 5; numDims++)
	{
		for (std::size_t t = 0; t != numTrials; ++t) {
			// create random objective vectors
			std::vector<Individual<RealVector,RealVector> > population1(numPoints);
			for (std::size_t i = 0; i != numPoints; ++i) {
				population1[i].penalizedFitness().resize(numDims);
				for (std::size_t j = 0; j != numDims; ++j) {
					population1[i].penalizedFitness()[j] = Rng::uni(-1,2);
					// make sure that some values coincide
					if (Rng::coinToss()) population1[i].penalizedFitness()[j] = std::round(population1[i].penalizedFitness()[j]);
				}
			}
			std::vector<Individual<RealVector,RealVector> > population2 = population1;

			FastNonDominatedSort sorter1;
			sorter1(population1);

			DCNonDominatedSort sorter2;
			sorter2(population2);

			// check that ranks are consistent with the dominance relation
			FitnessExtractor e;
			for(std::size_t i = 0; i != numPoints; ++i){
				for(std::size_t j = 0; j != numPoints; ++j){
					DominanceRelation rel = dominance(e(population1[i]), e(population1[j]));
					if (rel == LHS_DOMINATES_RHS) {
						BOOST_CHECK(population1[i].rank() < population1[j].rank());
					} else if (rel == RHS_DOMINATES_LHS) {
						BOOST_CHECK(population1[i].rank() > population1[j].rank());
					}

					if (population1[i].rank() == population1[j].rank()) {
						BOOST_CHECK(rel == INCOMPARABLE || rel == EQUIVALENT);
					}
				}
			}

			// check that the sorting results coincide
			for (std::size_t i=0; i<numPoints; i++)
			{
				BOOST_CHECK_EQUAL(population1[i].rank(), population2[i].rank());
			}
		}
	}
}
BOOST_AUTO_TEST_SUITE_END()
