#define BOOST_TEST_MODULE DirectSearch_CMA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/CMA.h>
#include <shark/Algorithms/DirectSearch/OnePlusOneES.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Cigar.h>
#include <shark/ObjectiveFunctions/Benchmarks/Discus.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>

#include "../testFunction.h"

using namespace shark;

BOOST_AUTO_TEST_CASE( OnePlusOneES_Cigar )
{
	Cigar function(3);
	OnePlusOneES onePlusOneES;

	// std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	test_function( onePlusOneES, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_CASE( OnePlusOneES_Discus )
{
	Discus function(3);
	OnePlusOneES onePlusOneES;

	// std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	test_function( onePlusOneES, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}


BOOST_AUTO_TEST_CASE( OnePlusOneES_Ellipsoid )
{
	Ellipsoid function(5);
	OnePlusOneES onePlusOneES;

	//std::cout << "Testing: " << optimizer.name() << " with " << function.name() << std::endl;
	test_function( onePlusOneES, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}
