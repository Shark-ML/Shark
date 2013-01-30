#include "shark/LinAlg/arrayoptimize.h"


#define BOOST_TEST_MODULE LinAlg_lnsrch
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

// Function to be decreased, implements f(x) = x*x
double my_func(const RealVector &x)
{
	return x(0) * x(0);
}

// Derivative function f'(x) = 2x of f(x) = x*x
//
double gradient(double x)
{
	return 2*x;
}


BOOST_AUTO_TEST_CASE( LinAlg_lnsrch )
{

	RealVector x_old(1);   // old point
	RealVector x_new(1);   // new point
	RealVector p(1);       // Newton direction
	RealVector grad(1);    // gradient of function at old point
	double     f_old(0.);  // function-value of old point
	double     f_new(0.);  // function-value of new point
	double     no_iter(1.);// no. of iterations, that
						   // "lnsrch" will do internally
	bool            status(false);  // status of "lnsrch"-function
	unsigned        i;              // number of "lnsrch" iterations

	// Initialize values for first call of "lnsrch":
	//
	x_old(0) = -9;
	f_old = my_func(x_old);       // Calculate function value of starting point
	grad(0) = gradient(x_old(0)); // Calculate gradient of starting point
	p(0) = 10.;                   // decrease function by going to
	                              // the "right" on the X-axis.
	x_new(0) = x_old(0);          // No new values calculated yet
	f_new = f_old;

	// Take 15 "lnsrch" iterations:
	for (i = 0; i < 14; i++)
	{
		// Calculate next new point:
		lnsrch(x_old, f_old, grad, p, x_new, f_new, no_iter, status, my_func);

		// Setting values for next iteration:
		x_old(0) = x_new(0);
		f_old = f_new;
		grad(0) = gradient(x_old(0));
	}

	BOOST_CHECK_SMALL(f_new,1.e-14);
}

