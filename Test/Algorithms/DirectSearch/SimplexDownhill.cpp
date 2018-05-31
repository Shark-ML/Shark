#define BOOST_TEST_MODULE DirectSearch_SimplexDownhill
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/SimplexDownhill.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>
#include <shark/ObjectiveFunctions/Benchmarks/Sphere.h>

#include "../testFunction.h"

using namespace shark;
using namespace shark::blas;
using namespace shark::benchmarks;


class TestObjective : public SingleObjectiveFunction
{
public:
	TestObjective()
	{
		m_features |= HAS_VALUE;
		m_features |= CAN_PROPOSE_STARTING_POINT;
	}

	std::size_t numberOfVariables() const
	{ return 2; }

	RealVector proposeStartingPoint() const
	{ return RealVector(2, 0.0); }

	double eval(RealVector const& x) const
	{
		double value = 0.0;
		double best = 1e100;
		for (size_t i=0; i<m_point.size(); i++)
		{
			double d = norm_2(x - m_point[i]);
			if (d < best)
			{
				best = d;
				value = m_value[i];
			}
		}
		return value;
	}

	std::vector<RealVector> m_point;
	std::vector<double> m_value;
};


BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_SimplexDownhill)

BOOST_AUTO_TEST_CASE( SimplexDownhill_Sphere )
{
	Sphere function(3);
	SimplexDownhill optimizer;

	std::cout<<"\nTesting: "<<optimizer.name()<<" with "<<function.name()<<std::endl;
	testFunction( optimizer, function, 10, 200, 1E-10 );
}

BOOST_AUTO_TEST_CASE( SimplexDownhill_Ellipsoid )
{
	Ellipsoid function(5);
	SimplexDownhill optimizer;

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	testFunction( optimizer, function, 10, 400, 1E-10 );
}

BOOST_AUTO_TEST_CASE( SimplexDownhill_Rosenbrock )
{
	Rosenbrock function( 3 );
	SimplexDownhill optimizer;

	std::cout << "\nTesting: " << optimizer.name() << " with " << function.name() << std::endl;
	testFunction( optimizer, function, 10, 500, 1E-10 );
}

BOOST_AUTO_TEST_CASE( SimplexDownhill_Reflection )
{
	TestObjective function;
	RealVector p0 = {+1.0, -0.5};
	RealVector p1 = {-0.5, +1.0};
	RealVector p2 = {-0.5, -0.5};
	function.m_point.push_back(p0); function.m_value.push_back(0.0);
	function.m_point.push_back(p1); function.m_value.push_back(1.0);
	function.m_point.push_back(p2); function.m_value.push_back(2.0);
	RealVector x0 = 0.5 * (p0 + p1);
	RealVector xr = 2.0 * x0 - p2;
	function.m_point.push_back(xr); function.m_value.push_back(0.5);

	SimplexDownhill optimizer;
	std::cout << "\nTesting: " << optimizer.name() << " reflection step" << std::endl;
	optimizer.init(function);
	optimizer.step(function);

	double deviation = norm_2(xr - optimizer.simplex()[2].point);
	BOOST_CHECK_SMALL(deviation, 1e-10);
}

BOOST_AUTO_TEST_CASE( SimplexDownhill_Expansion_1 )
{
	TestObjective function;
	RealVector p0 = {+1.0, -0.5};
	RealVector p1 = {-0.5, +1.0};
	RealVector p2 = {-0.5, -0.5};
	function.m_point.push_back(p0); function.m_value.push_back(0.0);
	function.m_point.push_back(p1); function.m_value.push_back(1.0);
	function.m_point.push_back(p2); function.m_value.push_back(2.0);
	RealVector x0 = 0.5 * (p0 + p1);
	RealVector xr = 2.0 * x0 - p2;
	RealVector xe = 3.0 * x0 - 2.0 * p2;
	function.m_point.push_back(xr); function.m_value.push_back(-1.0);
	function.m_point.push_back(xe); function.m_value.push_back(-2.0);

	SimplexDownhill optimizer;
	std::cout << "\nTesting: " << optimizer.name() << " expansion step 1" << std::endl;
	optimizer.init(function);
	optimizer.step(function);

	double deviation = norm_2(xe - optimizer.simplex()[2].point);
	BOOST_CHECK_SMALL(deviation, 1e-10);
}

BOOST_AUTO_TEST_CASE( SimplexDownhill_Expansion_2 )
{
	TestObjective function;
	RealVector p0 = {+1.0, -0.5};
	RealVector p1 = {-0.5, +1.0};
	RealVector p2 = {-0.5, -0.5};
	function.m_point.push_back(p0); function.m_value.push_back(0.0);
	function.m_point.push_back(p1); function.m_value.push_back(1.0);
	function.m_point.push_back(p2); function.m_value.push_back(2.0);
	RealVector x0 = 0.5 * (p0 + p1);
	RealVector xr = 2.0 * x0 - p2;
	RealVector xe = 3.0 * x0 - 2.0 * p2;
	function.m_point.push_back(xr); function.m_value.push_back(-2.0);
	function.m_point.push_back(xe); function.m_value.push_back(-1.0);

	SimplexDownhill optimizer;
	std::cout << "\nTesting: " << optimizer.name() << " expansion step 2" << std::endl;
	optimizer.init(function);
	optimizer.step(function);

	double deviation = norm_2(xr - optimizer.simplex()[2].point);
	BOOST_CHECK_SMALL(deviation, 1e-10);
}

BOOST_AUTO_TEST_CASE( SimplexDownhill_Contraction )
{
	TestObjective function;
	RealVector p0 = {+1.0, -0.5};
	RealVector p1 = {-0.5, +1.0};
	RealVector p2 = {-0.5, -0.5};
	function.m_point.push_back(p0); function.m_value.push_back(0.0);
	function.m_point.push_back(p1); function.m_value.push_back(1.0);
	function.m_point.push_back(p2); function.m_value.push_back(2.0);
	RealVector x0 = 0.5 * (p0 + p1);
	RealVector xr = 2.0 * x0 - p2;
	RealVector xc = 0.5 * x0 + 0.5 * p2;
	function.m_point.push_back(xr); function.m_value.push_back(1.5);
	function.m_point.push_back(xc); function.m_value.push_back(1.8);

	SimplexDownhill optimizer;
	std::cout << "\nTesting: " << optimizer.name() << " contraction step" << std::endl;
	optimizer.init(function);
	optimizer.step(function);

	double deviation = norm_2(xc - optimizer.simplex()[2].point);
	BOOST_CHECK_SMALL(deviation, 1e-10);
}

BOOST_AUTO_TEST_CASE( SimplexDownhill_Reduction )
{
	TestObjective function;
	RealVector p0 = {+1.0, -0.5};
	RealVector p1 = {-0.5, +1.0};
	RealVector p2 = {-0.5, -0.5};
	function.m_point.push_back(p0); function.m_value.push_back(0.0);
	function.m_point.push_back(p1); function.m_value.push_back(1.0);
	function.m_point.push_back(p2); function.m_value.push_back(2.0);
	RealVector x0 = 0.5 * (p0 + p1);
	RealVector xr = 2.0 * x0 - p2;
	RealVector xc = 0.5 * x0 + 0.5 * p2;
	function.m_point.push_back(xr); function.m_value.push_back(1.5);
	function.m_point.push_back(xc); function.m_value.push_back(3.0);

	SimplexDownhill optimizer;
	std::cout << "\nTesting: " << optimizer.name() << " reduction step" << std::endl;
	optimizer.init(function);
	optimizer.step(function);

	double deviation = norm_2(0.5 * (p0 + p1) - optimizer.simplex()[1].point)
	                 + norm_2(0.5 * (p0 + p2) - optimizer.simplex()[2].point);
	BOOST_CHECK_SMALL(deviation, 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()
