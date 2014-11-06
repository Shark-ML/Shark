#define BOOST_TEST_MODULE ML_NoisyRProp
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <shark/Rng/GlobalRng.h>
#include <shark/Algorithms/GradientDescent/NoisyRprop.h>

using namespace shark;

struct TestFunction : public SingleObjectiveFunction
{
	typedef SingleObjectiveFunction Base;

	RealMatrix A;
	bool m_noisy;
	TestFunction(bool noisy):A(3,3),m_noisy(noisy)
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
	virtual double eval(RealVector const& pattern)const
	{
		return inner_prod(prod(A,pattern),pattern);
	}
	//noise is only applied on the gradient
	virtual double evalDerivative(RealVector const& pattern, FirstOrderDerivative& derivative)const
	{
		derivative = 2*prod(A,pattern);
		if(m_noisy){
			derivative(0)+=Rng::gauss(0,0.001);
			derivative(1)+=Rng::gauss(0,0.001);
			derivative(2)+=Rng::gauss(0,0.001);
		}
		return eval(pattern);
	}
};

//First Test: the algorithm should work in a deterministic setting
BOOST_AUTO_TEST_SUITE (Algorithms_GradientDescent_NoisyRprop)

BOOST_AUTO_TEST_CASE( NoisyRprop_deterministic )
{
	TestFunction function(false);
	RealVector start(3);//startingPoint
	start(0)=2;
	start(1)=2;
	start(2)=2;
	NoisyRprop optimizer;
	optimizer.init(function,start);

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	double error=0;
	for(size_t iteration=0;iteration<300000;++iteration)
	{
		optimizer.step(function);
		if(iteration%10000==0)
		{
			error=optimizer.solution().value;
			RealVector best=optimizer.solution().point;
			std::cout<<iteration<<" error:"<<error<<std::endl;
		}

	}
	BOOST_CHECK_SMALL(error,1.e-8);
}
//Noise should also work
BOOST_AUTO_TEST_CASE( NoisyRprop_stochastic )
{
	Rng::seed(42);
	TestFunction function(true);
	RealVector start(3);//startingPoint
	start(0)=2;
	start(1)=2;
	start(2)=2;
	NoisyRprop optimizer;
	optimizer.init(function,start);

	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	double error=0;
	for(size_t iteration=0;iteration<300000;++iteration)
	{
		optimizer.step(function);
		if(iteration%10000==0)
		{
			error=optimizer.solution().value;
			RealVector best=optimizer.solution().point;
			std::cout<<iteration<<" error:"<<error<<std::endl;
		}

	}
	BOOST_CHECK_SMALL(error,1.e-7);
}

BOOST_AUTO_TEST_SUITE_END()
