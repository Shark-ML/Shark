#define BOOST_TEST_MODULE DirectSearch_CMA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/CMA.h>
#include <shark/Algorithms/DirectSearch/ElitistCMA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Cigar.h>
#include <shark/ObjectiveFunctions/Benchmarks/Discus.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>
#include <shark/ObjectiveFunctions/Benchmarks/Sphere.h>

#include "../testFunction.h"

using namespace shark;

BOOST_AUTO_TEST_CASE( CMA_Cigar )
{
	Cigar function(3);
	CMA optimizer;
	ElitistCMA elitistCMA;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_CASE( CMA_Discus )
{
	Discus function(3);
	CMA optimizer;

	std::cout<<"\nTesting: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_CASE( CMA_Ellipsoid )
{
	Ellipsoid function(5);
	CMA optimizer;

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_CASE( CMA_Rosenbrock )
{
	Rosenbrock function( 3 );
	CMA optimizer;

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	test_function( optimizer, function, _trials = 10, _iterations = 1000, _epsilon = 1E-10 );
}

BOOST_AUTO_TEST_CASE( CMA__Ellipsoid_Niko )
{
	const unsigned N = 10;
	RealVector x0(10, 0.1);
	Ellipsoid elli(N, 1E6);

	CMA cma;
	cma.init(elli, x0);
	cma.setSigma(0.1);

	for(unsigned i=0; i<6000; i++) 	cma.step( elli );
	BOOST_CHECK(cma.solution().value < 1E-8);
	BOOST_CHECK(cma.condition() > 1E5);
}

BOOST_AUTO_TEST_CASE( CMA_Sphere_Niko )
{
	const unsigned N = 10;
	RealVector x0(10, 0.1);
	Sphere sphere(N);

	CMA cma;
	cma.init(sphere, x0);
	cma.setSigma(1e-4);

	bool sigmaHigh = false;
	bool condHigh = false;
	for(unsigned i=0; i<1500; i++) {
		cma.step( sphere );
		if(cma.sigma() > 0.01) sigmaHigh = true;
		if(cma.condition() > 10) condHigh = true;
	}
	BOOST_CHECK(cma.solution().value < 1E-9);
	BOOST_CHECK(sigmaHigh);
	BOOST_CHECK(!condHigh);
}
