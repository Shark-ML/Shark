#define BOOST_TEST_MODULE GradDesc_LBFGS
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/GradientDescent/LBFGS.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>

#include "../testFunction.h"

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_GradientDescent_LBFGS)

BOOST_AUTO_TEST_CASE( LBFGS_dlinmin )
{
	Ellipsoid function(5);
	LBFGS optimizer;
	optimizer.setHistCount(3);
	optimizer.lineSearch().lineSearchType()=LineSearch::Dlinmin;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and dlinmin"<<std::endl;
	testFunction(optimizer,function,100,100);
}
BOOST_AUTO_TEST_CASE( LBFGS_wolfe )
{
	Ellipsoid function(5);
	LBFGS optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearch::WolfeCubic;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and wolfe line search"<<std::endl;
	testFunction(optimizer,function,100,100);
}
BOOST_AUTO_TEST_CASE( LBFGS_Dlinmin_Rosenbrock )
{
	Rosenbrock function(3);
	LBFGS optimizer;
	optimizer.setHistCount(3);
	optimizer.lineSearch().lineSearchType()=LineSearch::Dlinmin;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and dlinmin"<<std::endl;
	testFunction(optimizer,function,100,100);
}
BOOST_AUTO_TEST_CASE( BFGS_wolfe_Rosenbrock )
{
	Rosenbrock function(3);
	LBFGS optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearch::WolfeCubic;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and wolfe line search"<<std::endl;
	testFunction(optimizer,function,100,100);
}

BOOST_AUTO_TEST_SUITE_END()
