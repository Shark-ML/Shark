#include "shark/LinAlg/arrayoptimize.h"

#define BOOST_TEST_MODULE LinAlg_linmin
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

// The testfunction:
double testfunc(const RealVector& x)
{
	return x(0) * x(0) + 2;
}


BOOST_AUTO_TEST_CASE( LinAlg_linmin )
{
	double          fret(0.); // function value at the point that
	// is found by the function
	RealVector p(1);     // initial search starting point
	RealVector xi(1);    // direction of search

	// Initialize search starting point and direction:
	p(0) = -3.;
	xi(0) = 3.;

	// Minimize function:
	linmin(p, xi, fret, testfunc);

	BOOST_CHECK_SMALL(fret-2,1.e-14);
}
