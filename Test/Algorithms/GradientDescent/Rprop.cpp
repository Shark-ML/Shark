#define BOOST_TEST_MODULE ML_RProp
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>

#include "../testFunction.h"

using namespace shark;
using namespace shark::benchmarks;

BOOST_AUTO_TEST_SUITE (Algorithms_GradientDescent_Rprop)

typedef boost::mpl::list<Ellipsoid, Rosenbrock > Functions;

BOOST_AUTO_TEST_CASE( RProp_Tests_Ellipsoid){
	Ellipsoid function(5);
	Rprop<> optimizer;
	
	std::cout<<"Testing: IRpropPlus"<<" with "<<function.name()<<"... "<<std::endl;
	testFunction(optimizer,function,100,1000);
	
	optimizer.setUseOldValue(false);
	std::cout<<"Testing: RpropPlus"<<" with "<<function.name()<<"... "<<std::endl;
	optimizer.init(function);
	testFunction(optimizer,function,100,1000);
	
	optimizer.setUseBacktracking(false);
	std::cout<<"Testing: IRpropMinus"<<" with "<<function.name()<<"... "<<std::endl;
	optimizer.init(function);	
	testFunction(optimizer,function,100,1000);
	
	optimizer.setUseFreezing(false);
	std::cout<<"Testing: RpropMinus"<<" with "<<function.name()<<"... "<<std::endl;
	optimizer.init(function);
	testFunction(optimizer,function,100,1000);
}
BOOST_AUTO_TEST_CASE( RProp_Tests_Rosenbrock){
	Rosenbrock function(5);
	Rprop<> optimizer;
	
	std::cout<<"Testing: IRpropPlus"<<" with "<<function.name()<<"... "<<std::endl;
	optimizer.init(function);
	testFunction(optimizer,function,100,100000);
	
	optimizer.setUseOldValue(false);
	std::cout<<"Testing: RpropPlus"<<" with "<<function.name()<<"... "<<std::endl;
	optimizer.init(function);
	testFunction(optimizer,function,100,100000);
	
	optimizer.setUseBacktracking(false);
	std::cout<<"Testing: IRpropMinus"<<" with "<<function.name()<<"... "<<std::endl;
	optimizer.init(function);
	testFunction(optimizer,function,100,100000);
	
	optimizer.setUseFreezing(false);
	std::cout<<"Testing: RpropMinus"<<" with "<<function.name()<<"... "<<std::endl;
	optimizer.init(function);
	testFunction(optimizer,function,100,100000);
	
}

BOOST_AUTO_TEST_SUITE_END()
