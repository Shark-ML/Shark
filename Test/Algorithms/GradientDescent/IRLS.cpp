#define BOOST_TEST_MODULE GradDesc_IRLS
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/GradientDescent/IRLS.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>

#include "../testFunction.h"

using namespace shark;


BOOST_AUTO_TEST_CASE( IRLS_dlinmin )
{
	Ellipsoid function(5);
	IRLS optimizer;
	optimizer.hessianIsPositiveDefinite(true);
	optimizer.lineSearch().lineSearchType()=LineSearch::Dlinmin;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and dlinmin"<<std::endl;
	testFunction(optimizer,function,100,1);
}
BOOST_AUTO_TEST_CASE( IRLS_Dlinmin_Rosenbrock )
{
	Rosenbrock function(3);
	IRLS optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearch::Dlinmin;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and dlinmin"<<std::endl;
	testFunction(optimizer,function,100,1000,1.e-14);
}

