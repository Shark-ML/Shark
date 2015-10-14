#define BOOST_TEST_MODULE GradDesc_BFGS
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/GradientDescent/BFGS.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>

#include "../testFunction.h"

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_GradientDescent_BFGS)

BOOST_AUTO_TEST_CASE( BFGS_dlinmin )
{
	Ellipsoid function(5);
	BFGS optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearch::Dlinmin;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and dlinmin"<<std::endl;
	testFunction(optimizer,function,100,100);
}
BOOST_AUTO_TEST_CASE( BFGS_WolfeCubic )
{
	Ellipsoid function(5);
	BFGS optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearch::WolfeCubic;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and WolfeCubic"<<std::endl;
	testFunction(optimizer,function,100,100);
}
BOOST_AUTO_TEST_CASE( BFGS_Dlinmin_Rosenbrock )
{
	Rosenbrock function(3);
	BFGS optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearch::Dlinmin;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and dlinmin"<<std::endl;
	testFunction(optimizer,function,100,2000);
}
BOOST_AUTO_TEST_CASE( BFGS_WolfeCubic_Rosenbrock )
{
	Rosenbrock function(3);
	BFGS optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearch::WolfeCubic;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and WolfeCubic"<<std::endl;
	testFunction(optimizer,function,100,2000);
}
BOOST_AUTO_TEST_SUITE_END()
