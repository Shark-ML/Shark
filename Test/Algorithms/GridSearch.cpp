#define BOOST_TEST_MODULE ML_GridSearch
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/GridSearch.h>

using namespace shark;

struct TestFunction : public SingleObjectiveFunction
{
	typedef SingleObjectiveFunction Base;

	RealMatrix A;
	TestFunction():A(2,2)
	{
		A.clear();
		A(0,0)=1;
		A(1,1)=1;

		m_features|=Base::HAS_FIRST_DERIVATIVE;
		m_features|=Base::CAN_PROPOSE_STARTING_POINT;
	}

	std::string name() const
	{ return "TestFunction"; }
	
	std::size_t numberOfVariables()const{
		return 2;
	}

	virtual double eval(RealVector const& pattern)const
	{
		return inner_prod(prod(A,pattern),pattern);
	}
	
	virtual void proposeStartingPoint( SearchPointType & startingPoint /* IN & OUT */ )const {
		startingPoint.resize(2);
		startingPoint.clear();
	}
};

BOOST_AUTO_TEST_CASE( NestedGridSearch_initialized )
{
	std::vector<double> searchMin;
	searchMin.push_back(-1);
	searchMin.push_back(-1);

	std::vector<double> searchMax;
	searchMax.push_back(1);
	searchMax.push_back(1);

	TestFunction function;
	NestedGridSearch optimizer;
	optimizer.configure(searchMin,searchMax);
	optimizer.init(function);

	// train the model
	double error=0;
	for(size_t iteration=0;iteration<30;++iteration)
	{
		optimizer.step(function);
		error=optimizer.solution().value;
	}
	std::cout<<"NestedGridSearch done. Error:"<<error<<std::endl;
	BOOST_CHECK_SMALL(error,1.e-15);
}
BOOST_AUTO_TEST_CASE( NestedGridSearch_uninitialized )
{
	std::vector<double> searchMin;
	searchMin.push_back(-1);
	searchMin.push_back(-1);

	std::vector<double> searchMax;
	searchMax.push_back(1);
	searchMax.push_back(1);

	TestFunction function;
	NestedGridSearch optimizer;
	optimizer.init(function);

	// train the model
	double error=0;
	for(size_t iteration=0;iteration<30;++iteration)
	{
		optimizer.step(function);
		error=optimizer.solution().value;
	}
	std::cout<<"NestedGridSearch_uninitialized done. Error:"<<error<<std::endl;
	BOOST_CHECK_SMALL(error,1.e-15);
}
BOOST_AUTO_TEST_CASE( GridSearch_init_uniform )
{
	TestFunction function;
	GridSearch optimizer;
	optimizer.configure(2,-1,1,5);
	optimizer.init(function);

	// train the model
	optimizer.step(function);
	double error=optimizer.solution().value;
	std::cout<<"GridSearch_init_uniform done. Error:"<<error<<std::endl;
	BOOST_CHECK_SMALL(error,1.e-15);
}
BOOST_AUTO_TEST_CASE( GridSearch_uninitialized )
{
	TestFunction function;
	GridSearch optimizer;
	optimizer.init(function);

	// train the model
	optimizer.step(function);
	double error=optimizer.solution().value;
	std::cout<<"GridSearch_uninitialized done. Error:"<<error<<std::endl;
	BOOST_CHECK_SMALL(error,1.e-15);
}
BOOST_AUTO_TEST_CASE( GridSearch_init_individual )
{
	std::vector<double> searchMin;
	searchMin.push_back(-1);
	searchMin.push_back(-1);

	std::vector<double> searchMax;
	searchMax.push_back(1);
	searchMax.push_back(1);

	std::vector<size_t> sections;
	sections.push_back(5);
	sections.push_back(5);

	TestFunction function;
	GridSearch optimizer;
	optimizer.configure(searchMin,searchMax,sections);
	optimizer.init(function);
	// train the model
	optimizer.step(function);
	double error=optimizer.solution().value;
	std::cout<<"GridSearch_init_individual done. Error:"<<error<<std::endl;
	BOOST_CHECK_SMALL(error,1.e-15);
}
BOOST_AUTO_TEST_CASE( GridSearch_init_individual_param )
{
	std::vector<std::vector<double> > searchValues(2);
	for(size_t i=0;i!=2;++i)
	{
		for(size_t j=0;j<5;++j)
		{
			searchValues[i].push_back(-1+j*0.5);
		}
	}
	TestFunction function;
	GridSearch optimizer;
	optimizer.configure(searchValues);
	optimizer.init(function);
	// train the model
	optimizer.step(function);
	double error=optimizer.solution().value;
	std::cout<<"GridSearch_init_individual_param done. Error:"<<error<<std::endl;
	BOOST_CHECK_SMALL(error,1.e-15);
}
BOOST_AUTO_TEST_CASE( GridSearch_init_Model_Linear )
{
	TestFunction function;
	GridSearch optimizer;
	optimizer.init(function);
	optimizer.assignLinearRange(0, 5, -1, 1);
	optimizer.assignLinearRange(1, 5, -1, 1);
	// train the model
	optimizer.step(function);
	double error=optimizer.solution().value;
	std::cout<<"GridSearch_init_individual_param done. Error:"<<error<<std::endl;
	BOOST_CHECK_SMALL(error,1.e-15);
}
BOOST_AUTO_TEST_CASE( PointSearch_initialized )
{
	std::vector<RealVector> points;
	points.resize(11,RealVector(2));
	for(size_t i=0;i!=10;++i)
	{
		for(size_t j=0;j<2;++j)
		{
			points[i](j)=Rng::gauss(0,1);
		}
	}
	points[10](0)=0.0;
	points[10](1)=0.0;
	TestFunction function;
	PointSearch optimizer;
	optimizer.configure(points);
	optimizer.init(function);
	// train the model
	optimizer.step(function);
	double error=optimizer.solution().value;
	std::cout<<"PointSearch done. Error:"<<error<<std::endl;
	BOOST_CHECK_SMALL(error,1.e-15);
}
BOOST_AUTO_TEST_CASE( PointSearch_random )
{
	TestFunction function;
	PointSearch optimizer;
	//i hope that this enough, that the result will never fail...
	optimizer.configure(2,1000000,-0.1,0.1);
	optimizer.init(function);
	// train the model
	optimizer.step(function);
	double error=optimizer.solution().value;
	std::cout<<"PointSearch_random done. Error:"<<error<<std::endl;
	BOOST_CHECK_SMALL(error,1.e-5);
}
