#define BOOST_TEST_MODULE GradDesc_TrustRegionNewton
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/GradientDescent/TrustRegionNewton.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>

#include "../testFunction.h"

using namespace shark;
using namespace shark::benchmarks;


BOOST_AUTO_TEST_SUITE (Algorithms_GradientDescent_TrustRegionNewton)

BOOST_AUTO_TEST_CASE( TrustRegionNewton_Ellipsoid )
{
	Ellipsoid function(5);
	TrustRegionNewton optimizer;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	testFunction(optimizer,function,100,100);
}
BOOST_AUTO_TEST_CASE( TrustRegionNewton_Rosenbrock )
{
	Rosenbrock function(3);
	TrustRegionNewton optimizer;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	testFunction(optimizer,function,100,1000,1.e-14);
}


BOOST_AUTO_TEST_SUITE_END()
