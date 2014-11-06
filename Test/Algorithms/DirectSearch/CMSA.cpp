#define BOOST_TEST_MODULE DirectSearch_CMSA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/CMSA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Cigar.h>
#include <shark/ObjectiveFunctions/Benchmarks/Discus.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>

#include "../testFunction.h"

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_CMSA)

BOOST_AUTO_TEST_CASE( CMSA_Cigar )
{
	Cigar function(3);
	CMSA optimizer;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_CASE( CMSA_Discus )
{
	Discus function(3);
	CMSA optimizer;

	std::cout<<"\nTesting: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_CASE( CMSA_Ellipsoid )
{
	Ellipsoid function(5);
	CMSA optimizer;

	std::cout<<"\nTesting: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}
BOOST_AUTO_TEST_CASE( CMSA_Rosenbrock )
{
	Rosenbrock function( 3 );
	CMSA optimizer;

	std::cout<<"\nTesting: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	// TODO: Results here do not correspond to results in Beyer's paper.
	test_function( optimizer, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_SUITE_END()
