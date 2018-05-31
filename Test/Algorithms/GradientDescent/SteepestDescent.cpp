#define BOOST_TEST_MODULE ML_SteepestDescent
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/GradientDescent/SteepestDescent.h>

using namespace shark;


template<class VectorType>
struct TestFunction : public AbstractObjectiveFunction<VectorType, double>{

	RealMatrix A;
	TestFunction():A(3,3,0.0){

		A(0,0)=10;
		A(1,1)=5;
		A(2,2)=1;
		A(1,0)=1;
		A(0,1)=1;
		A(2,0)=1;
		A(0,2)=1;

		this->m_features |= this->HAS_FIRST_DERIVATIVE;
	}

	std::string name() const
	{ return "TestFunction"; }

	std::size_t numberOfVariables()const{
		return 3;
	}

	virtual double eval(VectorType const& pattern)const
	{
		return inner_prod(A % pattern,pattern);
	}
	virtual double evalDerivative(VectorType const& pattern, VectorType& derivative)const
	{
		derivative = 2*A % pattern;
		return eval(pattern);
	}
};


BOOST_AUTO_TEST_SUITE (Algorithms_GradientDescent_SteepestDescent)

typedef boost::mpl::list<RealVector, FloatVector > VectorTypes;

BOOST_AUTO_TEST_CASE_TEMPLATE(SteepestDescent_Test, VectorType,VectorTypes){
	TestFunction<VectorType> function;
	VectorType start(3);//startingPoint
	start(0)=1;
	start(1)=1;
	start(2)=1;
	SteepestDescent<VectorType> optimizer;
	optimizer.setLearningRate(0.1*(1-0.3));
	optimizer.setMomentum(0.3);
	optimizer.init(function,start);


	// train the model
	std::cout<<"Testing: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	double error=0;
	for(size_t iteration=0;iteration<100;++iteration)
	{
		optimizer.step(function);
		error=optimizer.solution().value;
		VectorType best=optimizer.solution().point;
		std::cout<<iteration<<" error:"<<error<<" parameter:"<<best<<std::endl;
	}
	BOOST_CHECK_SMALL(error,1.e-15);
}

BOOST_AUTO_TEST_SUITE_END()
