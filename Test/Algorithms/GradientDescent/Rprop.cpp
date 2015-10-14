#define BOOST_TEST_MODULE ML_RProp
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>

#include "../testFunction.h"

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_GradientDescent_Rprop)

BOOST_AUTO_TEST_CASE( RPropPlus_Simple )
{
	Ellipsoid function(5);
	RpropPlus optimizer;
	optimizer.init(function);

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<"... "<<std::endl;
	testFunction(optimizer,function,100,1000);
}
BOOST_AUTO_TEST_CASE( RPropMinus_Simple )
{
	Ellipsoid function(5);
	RpropMinus optimizer;
	optimizer.init(function);

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<"... "<<std::endl;
	testFunction(optimizer,function,100,1000);
}
BOOST_AUTO_TEST_CASE( IRPropPlus_Simple )
{
	Ellipsoid function(5);
	IRpropPlus optimizer;
	optimizer.init(function);

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<"... "<<std::endl;
	testFunction(optimizer,function,100,1000);
}
BOOST_AUTO_TEST_CASE( IRPropMinus_Simple )
{
	Ellipsoid function(5);
	IRpropMinus optimizer;
	optimizer.init(function);

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<"... "<<std::endl;
	testFunction(optimizer,function,100,10000);
}
BOOST_AUTO_TEST_CASE( RPropPlus_Rosenbrock )
{
	Rosenbrock function(3);
	RpropPlus optimizer;
	optimizer.init(function);


	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<"... "<<std::endl;
	testFunction(optimizer,function,100,100000);
}
BOOST_AUTO_TEST_CASE( RPropMinus_Rosenbrock )
{
	Rosenbrock function(3);
	RpropMinus optimizer;
	optimizer.init(function);


	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<"... "<<std::endl;
	testFunction(optimizer,function,100,100000);
}
BOOST_AUTO_TEST_CASE( IRPropPlus_Rosenbrock )
{
	Rosenbrock function(3);
	IRpropPlus optimizer;
	optimizer.init(function);

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<"... "<<std::endl;
	testFunction(optimizer,function,100,10000);
}
BOOST_AUTO_TEST_CASE( IRPropMinus_Rosenbrock )
{
	Rosenbrock function(3);
	IRpropMinus optimizer;
	optimizer.init(function);


	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<"... "<<std::flush;
	testFunction(optimizer,function,100,100000);
}

BOOST_AUTO_TEST_SUITE_END()
