#define BOOST_TEST_MODULE DirectSearch_CrossEntropy
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/CrossEntropyMethod.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Cigar.h>
#include <shark/ObjectiveFunctions/Benchmarks/Discus.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>
#include <shark/ObjectiveFunctions/Benchmarks/Sphere.h>
#include <shark/ObjectiveFunctions/Benchmarks/DiffPowers.h>

#include "../testFunction.h"

using namespace shark;
using namespace shark::benchmarks;

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_CrossEntropyMethod)

BOOST_AUTO_TEST_CASE( CrossEntropyMethod_Cigar )
{
	Cigar function(3);
	CrossEntropyMethod optimizer;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	testFunction( optimizer, function, 10, 1000, 1E-10 );
}

BOOST_AUTO_TEST_CASE( CrossEntropyMethod_Discus )
{
	Discus function(3);
	CrossEntropyMethod optimizer;

	std::cout<<"\nTesting: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	testFunction( optimizer, function, 10, 1000, 1E-10 );
}

BOOST_AUTO_TEST_CASE( CrossEntropyMethod_Ellipsoid )
{
	Ellipsoid function(5);
	CrossEntropyMethod optimizer;

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	testFunction( optimizer, function, 10, 1000, 1E-10 );
}

BOOST_AUTO_TEST_CASE( CrossEntropyMethod_DiffPowers )
{
	DiffPowers function( 3 );
	CrossEntropyMethod optimizer;

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	testFunction( optimizer, function, 10, 10000, 1E-10 );
}


BOOST_AUTO_TEST_SUITE_END()
