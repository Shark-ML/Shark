#define BOOST_TEST_MODULE DirectSearch_Recombination

#include <boost/range/algorithm/equal.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/BitflipMutator.h>

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_Operators_Mutation_BitflipMutation)

BOOST_AUTO_TEST_CASE( BitflipMutation ) {
	std::size_t n = 10;
	std::size_t trials = 1000;
	shark::BitflipMutator flip;
	flip.m_mutationStrength = 0.3;
	
	std::vector<unsigned int> flipCount(n,0);
	
	shark::Individual< std::vector< bool >,double > ind1, ind2;
	ind1.searchPoint() = std::vector< bool >( n, false );
	ind2.searchPoint() = ind1.searchPoint();
	for(std::size_t t = 0; t != trials; ++t){
		flip(ind2);
		for(std::size_t i = 0; i != n; ++i){
			flipCount[i] += (ind1.searchPoint()[i] == ind2.searchPoint()[i])? 0:1;
		}
		ind1 = ind2;
	}	
	//check that not too many were flipped
	unsigned int minFlip = (unsigned int)(0.25*trials);
	unsigned int maxFlip = (unsigned int)(0.35*trials);
	for(std::size_t i = 0; i != n; ++i){
		BOOST_CHECK(flipCount[i] >= minFlip);
		BOOST_CHECK(flipCount[i] <= maxFlip);
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
