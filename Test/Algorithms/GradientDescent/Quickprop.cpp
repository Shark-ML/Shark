#define BOOST_TEST_MODULE ML_Quickprop
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/GradientDescent/Quickprop.h>

using namespace shark;

struct TestFunction : public SingleObjectiveFunction
{
	typedef SingleObjectiveFunction Base;

	RealMatrix A;
	TestFunction():A(3,3)
	{
		A.clear();
		A(0,0)=20;
		A(1,1)=10;
		A(2,2)=5;

		m_features|=Base::HAS_FIRST_DERIVATIVE;
	}

	std::string name() const
	{ return "TestFunction"; }

	std::size_t numberOfVariables()const{
		return 3;
	}
	
	// adds just a value c on the input
	virtual double eval(RealVector const& pattern)const
	{
		return inner_prod(prod(A,pattern),pattern);
	}
	virtual double evalDerivative(RealVector const& pattern, FirstOrderDerivative& derivative)const
	{
		derivative = 2*prod(A,pattern);
		return eval(pattern);
	}
};


BOOST_AUTO_TEST_CASE( Quickprop_Test )
{
	TestFunction function;
	RealVector start(3);//startingPoint
	start(0)=1;
	start(1)=1;
	start(2)=1;
	Quickprop optimizer;
	optimizer.init(function,start);


	// train the model
	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	double error=0;
	//3 iterations are completly sufficient! if it needs more, something is terribly wrong!
	for(size_t iteration=0;iteration<3;++iteration)
	{
		optimizer.step(function);
		error=optimizer.solution().value;
		RealVector best=optimizer.solution().point;
		std::cout<<iteration<<" error:"<<error<<" parameter:"<<best<<std::endl;
	}
	BOOST_CHECK_SMALL(error,1.e-15);
}
BOOST_AUTO_TEST_CASE( QuickpropOriginal_Test )
{
	TestFunction function;
	RealVector start(3);//startingPoint
	start(0)=1;
	start(1)=1;
	start(2)=1;
	QuickpropOriginal optimizer;
	optimizer.init(function,start);


	// train the model
	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	double error=0;
	for(size_t iteration=0;iteration<3;++iteration)
	{
		optimizer.step(function);
		error=optimizer.solution().value;
		RealVector best=optimizer.solution().point;
		std::cout<<iteration<<" error:"<<error<<" parameter:"<<best<<std::endl;
	}
	BOOST_CHECK_SMALL(error,1.e-15);
}

