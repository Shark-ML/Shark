#define BOOST_TEST_MODULE DirectSearch_Recombination
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Recombination/UniformCrossover.h>
#include <shark/LinAlg/Base.h>

BOOST_AUTO_TEST_CASE( UniformCrossover ) {


    shark::RealVector v1( 10, 1 ), v2( 10, 0 );

    shark::UniformCrossover uc;
    BOOST_CHECK_CLOSE( uc.mixingRatio(), 0.5, 1E-10 );
    BOOST_CHECK_NO_THROW( uc.mixingRatio() = 0.7 );
    BOOST_CHECK_CLOSE( uc.mixingRatio(), 0.7, 1E-10 );
    BOOST_CHECK_NO_THROW( uc.mixingRatio() = 0.5 );

    shark::RealVector v3;
    BOOST_CHECK_NO_THROW( v3 = uc( v1, v2 ) );    
}
