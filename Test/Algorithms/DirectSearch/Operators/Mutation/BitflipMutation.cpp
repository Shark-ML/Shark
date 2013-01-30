#define BOOST_TEST_MODULE DirectSearch_Recombination

#include <boost/range/algorithm/equal.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

#include <shark/Algorithms/DirectSearch/TypedIndividual.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/BitflipMutator.h>

BOOST_AUTO_TEST_CASE( BitflipMutation ) {
    std::size_t n = 1000;
    shark::BitflipMutator flip( 0.5 );

    shark::TypedIndividual< std::vector< bool > > ind1, ind2;
    *ind1 = std::vector< bool >( n, false );
    *ind2 = *ind1;

	BOOST_CHECK_NO_THROW( flip( ind2 ) );
    BOOST_CHECK(!boost::range::equal(*ind1, *ind2)); // well, 1/(2^1000) probability of failure actually
}
