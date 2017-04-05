#define BOOST_TEST_MODULE DirectSearch_FastNonDominatedSort
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Domination/FastNonDominatedSort.h>
#include <shark/Algorithms/DirectSearch/Operators/Domination/DCNonDominatedSort.h>
#include <shark/Core/Random.h>
#include <shark/Core/Timer.h>

#include <cmath>
#include <cassert>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_NonDominatedSort)

// Create random pointss of individuals, sorts them,
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
			std::vector<RealVector> points(numPoints);
			for (std::size_t i = 0; i != numPoints; ++i) {
				points[i].resize(numDims);
				for (std::size_t j = 0; j != numDims; ++j) {
					points[i][j] = random::uni(random::globalRng,-1,2);
					// make sure that some values coincide
					if (random::coinToss(random::globalRng)) points[i][j] = std::round(points[i][j]);
				}
			}
			std::vector<unsigned int> ranks1(numPoints);
			fastNonDominatedSort(points, ranks1);
			
			std::vector<unsigned int> ranks2(numPoints);
			dcNonDominatedSort(points, ranks2);

			// check that ranks are consistent with the dominance relation
			for(std::size_t i = 0; i != numPoints; ++i){
				for(std::size_t j = 0; j != numPoints; ++j){
					DominanceRelation rel = dominance(points[i], points[j]);
					if (rel == LHS_DOMINATES_RHS) {
						BOOST_CHECK_LT(ranks1[i], ranks1[j]);
					} else if (rel == RHS_DOMINATES_LHS) {
						BOOST_CHECK_GT(ranks1[i], ranks1[j]);
					}
					if (ranks1[i] == ranks1[j]) {
						BOOST_CHECK(rel == INCOMPARABLE || rel == EQUIVALENT);
					}
				}
			}

			// check that the sorting results coincide
			for (std::size_t i=0; i<numPoints; i++)
			{
				BOOST_CHECK_EQUAL(ranks1[i], ranks2[i]);
			}
		}
	}
}
BOOST_AUTO_TEST_SUITE_END()
