#include <shark/Algorithms/GradientDescent/LineSearch.h>

#define BOOST_TEST_MODULE LinAlg_dlmin
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
using namespace shark;
// The testfunction:
struct Testfunction
{
	double eval(const RealVector& x)
	{
		return x(0) * x(0) + 2;
	}


	// The derivation of the testfunction:
	double evalDerivative(const RealVector& x, RealVector& y)
	{
		y(0) = 2 * x(0);
		return eval(x);
	}
};

BOOST_AUTO_TEST_SUITE (Algorithms_GradientDescent_LineSearch)

BOOST_AUTO_TEST_CASE( LineSearch_DLinmin )
{
	double fret(0.);// function value at the point that
					// is found by the function
	RealVector p(1);     // initial search starting point
	RealVector xi(1);    // direction of search
	// Initialize search starting point and direction:
	p(0) = -3.;
	xi(0) = 3.;
	Testfunction function;
	// Minimize function:
	detail::dlinmin(p, xi, fret, function,0.0,1.0);

	// lines below are for self-testing this example, please ignore
	BOOST_CHECK_SMALL(fret-2,1.e-14);
}

BOOST_AUTO_TEST_SUITE_END()
