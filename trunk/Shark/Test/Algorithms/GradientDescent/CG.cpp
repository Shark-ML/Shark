#define BOOST_TEST_MODULE ML_CG
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/GradientDescent/CG.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>

#include "../testFunction.h"

using namespace shark;


BOOST_AUTO_TEST_CASE( CG_dlinmin )
{
	Ellipsoid function(5);
	CG optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearch::Dlinmin;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and dlinmin"<<std::endl;
	testFunction(optimizer,function,100,100);
}
BOOST_AUTO_TEST_CASE( CG_WolfeCubic )
{
	Ellipsoid function(5);
	CG optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearch::WolfeCubic;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and WolfeCubic"<<std::endl;
	testFunction(optimizer,function,1,500);
}
BOOST_AUTO_TEST_CASE( CG_Dlinmin_Rosenbrock )
{
	Rosenbrock function(3);
	CG optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearch::Dlinmin;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and dlinmin"<<std::endl;
	testFunction(optimizer,function,100,3000,1.e-14);
}
BOOST_AUTO_TEST_CASE( CG_WolfeCubic_Rosenbrock )
{
	Rosenbrock function(3);
	CG optimizer;
	optimizer.lineSearch().lineSearchType()=LineSearch::WolfeCubic;

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<" and WolfeCubic"<<std::endl;
	testFunction(optimizer,function,100,2000);
}
