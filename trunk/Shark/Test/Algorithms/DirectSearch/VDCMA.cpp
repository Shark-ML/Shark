#define BOOST_TEST_MODULE DirectSearch_VDCMA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/VDCMA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>

#include "../testFunction.h"

using namespace shark;

BOOST_AUTO_TEST_CASE( CMA_Ellipsoid )
{
	Ellipsoid function(20);
	VDCMA optimizer;
	optimizer.setInitialSigma(2);

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 5000/optimizer.suggestLambda(20), _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_CASE( CMA_Rosenbrock )
{
	Rosenbrock function( 20,4 );
	VDCMA optimizer;
	optimizer.setInitialSigma(2);
	std::size_t lambda= optimizer.suggestLambda(20);

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	test_function( optimizer, function, _trials = 101, _iterations = 22000/lambda, _epsilon = 1E-10 );
}