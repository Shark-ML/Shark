#define BOOST_TEST_MODULE DirectSearch_FastNonDominatedSort
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Domination/ParetoDominance.h>
#include <shark/Core/Random.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_ParetoDominance)

// check the Pareto relation on predefined points
BOOST_AUTO_TEST_CASE( ParetoDominance_RelationsTest ) {
	RealVector p0(3);
	p0[0] = -1; p0[1] = 0; p0[2] = 1;
	RealVector p1(3);  // dominated by p0
	p1[0] = 1; p1[1] = 2; p1[2] = 3;
	RealVector p2(3);  // dominated by p0, but equal component
	p2[0] = -1; p2[1] = 2; p2[2] = 3;
	RealVector p3(3);  // incomparable to p0
	p3[0] = -2; p3[1] = 0; p3[2] = 2;

	BOOST_CHECK_EQUAL(dominance(p0, p0), EQUIVALENT);
	BOOST_CHECK_EQUAL(dominance(p0, p1), LHS_DOMINATES_RHS);
	BOOST_CHECK_EQUAL(dominance(p0, p2), LHS_DOMINATES_RHS);
	BOOST_CHECK_EQUAL(dominance(p1, p0), RHS_DOMINATES_LHS);
	BOOST_CHECK_EQUAL(dominance(p2, p0), RHS_DOMINATES_LHS);
	BOOST_CHECK_EQUAL(dominance(p0, p3), INCOMPARABLE);
	BOOST_CHECK_EQUAL(dominance(p3, p0), INCOMPARABLE);
}

// check the Pareto relation on random sets of objective vectors
BOOST_AUTO_TEST_CASE( ParetoDominance_Random_Test ) {
	std::size_t numPoints = 20;
	std::size_t numTrials = 10;
	std::size_t numDims = 3;
	for(std::size_t t = 0; t != numTrials; ++t){ 
		// random create points
		std::vector<RealVector > population(numPoints);
		for(std::size_t i = 0; i != numPoints; ++i){
			population[i].resize(numDims);
			for(std::size_t j = 0; j != numDims; ++j){
				population[i][j]= random::uni(random::globalRng,-1,2);
			}
		}

		// check that ranks are okay
		for(std::size_t i = 0; i != numPoints; ++i){
			for(std::size_t j = 0; j != numPoints; ++j){
				//test all 6 results
				DominanceRelation rel = dominance(population[i],population[j]);
				switch (rel)
				{
					case INCOMPARABLE:
					{
						bool lt = false, gt = false;
						for(std::size_t k = 0; k != numDims; ++k) {
							lt = lt || (population[i][k] < population[j][k]);
							gt = gt || (population[i][k] > population[j][k]);
						}
						BOOST_CHECK(lt && gt);
						break;
					}
					case LHS_DOMINATES_RHS:
					{
						bool ne = false;
						for(std::size_t k = 0; k != numDims; ++k) {
							BOOST_CHECK_LE(population[i][k], population[j][k]);
							ne = ne || (population[i][k] < population[j][k]);
						}
						BOOST_CHECK(ne);
						break;
					}
					case RHS_DOMINATES_LHS:
					{
						bool ne = false;
						for(std::size_t k = 0; k != numDims; ++k) {
							BOOST_CHECK_GE(population[i][k], population[j][k]);
							ne = ne || (population[i][k] > population[j][k]);
						}
						BOOST_CHECK(ne);
						break;
					}
					case EQUIVALENT:
					{
						for(std::size_t k = 0; k != numDims; ++k) {
							BOOST_CHECK_EQUAL(population[i][k], population[j][k]);
						}
						break;
					}
				}
			}
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
