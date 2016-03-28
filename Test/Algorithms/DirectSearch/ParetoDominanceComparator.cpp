#define BOOST_TEST_MODULE DirectSearch_FastNonDominatedSort
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/ParetoDominance.h>
#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Rng/GlobalRng.h>

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

	BOOST_CHECK(equivalent(p0, p0));
	BOOST_CHECK(! prec(p0, p0));
	BOOST_CHECK(preceq(p0, p0));
	BOOST_CHECK(! succ(p0, p0));
	BOOST_CHECK(succeq(p0, p0));
	BOOST_CHECK(! incomparable(p0, p0));

	BOOST_CHECK(! equivalent(p0, p1));
	BOOST_CHECK(prec(p0, p1));
	BOOST_CHECK(preceq(p0, p1));
	BOOST_CHECK(! succ(p0, p1));
	BOOST_CHECK(! succeq(p0, p1));
	BOOST_CHECK(! incomparable(p0, p1));

	BOOST_CHECK(! equivalent(p0, p2));
	BOOST_CHECK(prec(p0, p2));
	BOOST_CHECK(preceq(p0, p2));
	BOOST_CHECK(! succ(p0, p2));
	BOOST_CHECK(! succeq(p0, p2));
	BOOST_CHECK(! incomparable(p0, p2));

	BOOST_CHECK(! equivalent(p2, p0));
	BOOST_CHECK(! prec(p2, p0));
	BOOST_CHECK(! preceq(p2, p0));
	BOOST_CHECK(succ(p2, p0));
	BOOST_CHECK(succeq(p2, p0));
	BOOST_CHECK(! incomparable(p2, p0));

	BOOST_CHECK(! equivalent(p0, p3));
	BOOST_CHECK(! prec(p0, p3));
	BOOST_CHECK(! preceq(p0, p3));
	BOOST_CHECK(! succ(p0, p3));
	BOOST_CHECK(! succeq(p0, p3));
	BOOST_CHECK(incomparable(p0, p3));
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
				population[i][j]= Rng::uni(-1,2);
			}
		}

		// check that ranks are okay
		for(std::size_t i = 0; i != numPoints; ++i){
			for(std::size_t j = 0; j != numPoints; ++j){
				//test all 6 results
				ParetoRelation rel = dominance<RealVector, IdentityFitnessExtractor>(population[i],population[j]);
				if (prec(rel)) {
					bool ne = false;
					for(std::size_t k = 0; k != numDims; ++k) {
						BOOST_CHECK(population[i][k] <= population[j][k]);
						ne = ne || (population[i][k] < population[j][k]);
					}
					BOOST_CHECK(ne);
				}
				if (succ(rel)) {
					bool ne = false;
					for(std::size_t k = 0; k != numDims; ++k) {
						BOOST_CHECK(population[i][k] >= population[j][k]);
						ne = ne || (population[i][k] > population[j][k]);
					}
					BOOST_CHECK(ne);
				}
				if (preceq(rel)) {
					for(std::size_t k = 0; k != numDims; ++k) {
						BOOST_CHECK(population[i][k] <= population[j][k]);
					}
				}
				if (succeq(rel)) {
					for(std::size_t k = 0; k != numDims; ++k) {
						BOOST_CHECK(population[i][k] >= population[j][k]);
					}
				}
				if (equivalent(rel)) {
					for(std::size_t k = 0; k != numDims; ++k) {
						BOOST_CHECK(population[i][k] == population[j][k]);
					}
				}
				if (incomparable(rel)) {
					bool lt = false, gt = false;
					for(std::size_t k = 0; k != numDims; ++k) {
						lt = lt || (population[i][k] < population[j][k]);
						gt = gt || (population[i][k] > population[j][k]);
					}
					BOOST_CHECK(lt && gt);
				}
			}
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
