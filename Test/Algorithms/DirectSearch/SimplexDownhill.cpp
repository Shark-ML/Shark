#define BOOST_TEST_MODULE DirectSearch_SimplexDownhill
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/SimplexDownhill.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>
#include <shark/ObjectiveFunctions/Benchmarks/Sphere.h>

#include "../testFunction.h"

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_SimplexDownhill)

BOOST_AUTO_TEST_CASE( SimplexDownhill_Sphere )
{
	Sphere function(3);
	SimplexDownhill optimizer;

	std::cout<<"\nTesting: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 200, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_CASE( SimplexDownhill_Ellipsoid )
{
	Ellipsoid function(5);
	SimplexDownhill optimizer;

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 400, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_CASE( SimplexDownhill_Rosenbrock )
{
	Rosenbrock function( 3 );
	SimplexDownhill optimizer;

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 500, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_SUITE_END()
