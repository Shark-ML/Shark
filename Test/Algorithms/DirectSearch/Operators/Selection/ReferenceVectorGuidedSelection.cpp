#define BOOST_TEST_MODULE DirectSearch_Operators_ReferenceVectorGuidedSelection

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Selection/ReferenceVectorGuidedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Lattice.h>

#include <iostream>

using namespace shark;


BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_Operators_ReferenceVectorGuidedSelection)

BOOST_AUTO_TEST_CASE(populationPartition_correct)
{
	typedef ReferenceVectorGuidedSelection rv;
	// Three unit vectors
    RealMatrix refVecs(3, 2);
    refVecs(0, 0) = 0;                refVecs(0, 1) = 1;
    refVecs(1, 0) = 1 / std::sqrt(2); refVecs(1, 1) = 1 / std::sqrt(2);
    refVecs(2, 0) = 1;                refVecs(2, 1) = 0;
	
	RealMatrix fitness(6, 2);
	// group 0
	fitness(0, 0) = 0;                      fitness(0, 1) = 1;
	fitness(1, 0) = 0.1;                    fitness(1, 1) = 0.9;
	// group 1
	fitness(2, 0) = 1 / std::sqrt(2);       fitness(2, 1) = 1 / std::sqrt(2);
	fitness(3, 0) = 1 / std::sqrt(2) - 0.1; fitness(3, 1) = 1 / std::sqrt(2) + 0.1;
	// group 2
	fitness(4, 0) = 1;                      fitness(4, 1) = 0;
	fitness(5, 0) = 0.9;                    fitness(5, 1) = 0.1;

	const std::vector<std::set<std::size_t>> groups = rv::populationPartition(rv::cosAngles(fitness, refVecs));
	
	std::set<std::size_t> exp0{0, 1};
	std::set<std::size_t> exp1{2, 3};
	std::set<std::size_t> exp2{4, 5};
	BOOST_CHECK(groups[0] == exp0);
	BOOST_CHECK(groups[1] == exp1);
	BOOST_CHECK(groups[2] == exp2);
}


BOOST_AUTO_TEST_SUITE_END()
