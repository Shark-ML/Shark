#define BOOST_TEST_MODULE DirectSearch_CrossEntropy
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/CrossEntropy.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Cigar.h>
#include <shark/ObjectiveFunctions/Benchmarks/Discus.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>
#include <shark/ObjectiveFunctions/Benchmarks/Sphere.h>
#include <shark/ObjectiveFunctions/Benchmarks/DiffPowers.h>

#include "../testFunction.h"

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_CrossEntropy)

BOOST_AUTO_TEST_CASE( CrossEntropy_Cigar )
{
	Cigar function(3);
	CrossEntropy optimizer;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_CASE( CrossEntropy_Discus )
{
	Discus function(3);
	CrossEntropy optimizer;

	std::cout<<"\nTesting: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_CASE( CrossEntropy_Ellipsoid )
{
	Ellipsoid function(5);
	CrossEntropy optimizer;

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_CASE( CrossEntropy_DiffPowers )
{
	DiffPowers function( 3 );
	CrossEntropy optimizer;

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 10000, _epsilon = 1E-10 );
}


BOOST_AUTO_TEST_SUITE_END()
