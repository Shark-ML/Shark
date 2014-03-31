#define BOOST_TEST_MODULE DirectSearch_Recombination

#include <boost/range/algorithm/equal.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/BitflipMutator.h>

BOOST_AUTO_TEST_CASE( BitflipMutation ) {
	std::size_t n = 1000;
	shark::BitflipMutator flip( 0.5 );

	shark::Individual< std::vector< bool >,double > ind1, ind2;
	ind1.searchPoint() = std::vector< bool >( n, false );
	ind2.searchPoint() = ind1.searchPoint();

	BOOST_CHECK_NO_THROW( flip( ind2 ) );
	BOOST_CHECK(!boost::range::equal(ind1.searchPoint(), ind2.searchPoint())); // well, 1/(2^1000) probability of failure actually
}
