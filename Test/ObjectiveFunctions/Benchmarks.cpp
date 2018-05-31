#define BOOST_TEST_MODULE ObjectiveFunctions_Benchmarks
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/progress.hpp>
#include <boost/serialization/vector.hpp>

#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

#include <shark/Statistics/Statistics.h>

#include "TestObjectiveFunction.h"

BOOST_AUTO_TEST_SUITE (ObjectiveFunctions_Benchmarks)

BOOST_AUTO_TEST_CASE( Himmelblau ) {
	shark::benchmarks::Himmelblau hb;
	shark::RealVector v( 2 );
	
	v( 0 ) = -0.270844;
	v( 1 ) = -0.923038;

	BOOST_CHECK_SMALL( hb( v ) - 181.616, 1E-3 );

	v( 0 ) = 3;
	v( 1 ) = 2;
	BOOST_CHECK_SMALL( hb( v ), 1E-10 );

	v( 0 ) = -2.805118;
	v( 1 ) = 3.131312;
	BOOST_CHECK_SMALL( hb( v ), 1E-10 );

	v( 0 ) = -3.779310;
	v( 1 ) = -3.283186;
	BOOST_CHECK_SMALL( hb( v ), 1E-10 );

	v( 0 ) = 3.584428;
	v( 1 ) = -1.848126;
	BOOST_CHECK_SMALL( hb( v ), 1E-10 );
}

BOOST_AUTO_TEST_CASE( Rosenbrock_Derivative )
{
	const std::size_t dimensions = 5;
	const unsigned int trials = 10000;
	
	shark::benchmarks::Rosenbrock rosenbrock(dimensions);
	for(unsigned int i = 0; i != trials; ++i)
	{
		shark::RealVector point = rosenbrock.proposeStartingPoint();
		shark::testDerivative(rosenbrock, point,1.e-7,1.e-7,0.005);
	}
}
BOOST_AUTO_TEST_CASE( Ellipsoid_Derivative )
{
	const std::size_t dimensions = 5;
	const unsigned int trials = 10000;
	
	shark::benchmarks::Ellipsoid ellipsoid(dimensions);
	for(unsigned int i = 0; i != trials; ++i)
	{
		shark::RealVector point = ellipsoid.proposeStartingPoint();
		shark::testDerivative(ellipsoid, point,1.e-4,1.e-9);
	}
}

BOOST_AUTO_TEST_SUITE_END()
