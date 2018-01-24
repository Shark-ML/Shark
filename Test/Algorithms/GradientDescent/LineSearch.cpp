#include <shark/Algorithms/GradientDescent/LineSearch.h>

#define BOOST_TEST_MODULE LinAlg_dlmin
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
using namespace shark;
// The testfunction:
struct Testfunction: public SingleObjectiveFunction
{
	std::size_t numberOfVariables()const{return 0;}
	double eval(const RealVector& x)const
	{
		return x(0) * x(0) + 2;
	}


	// The derivation of the testfunction:
	double evalDerivative(const RealVector& x, RealVector& y)const
	{
		y(0) = 2 * x(0);
		return eval(x);
	}
};

BOOST_AUTO_TEST_SUITE (Algorithms_GradientDescent_LineSearch)

BOOST_AUTO_TEST_CASE( LineSearch_DLinmin )
{
	double fret = 11;// function value at the point that
					// is found by the function
	RealVector p(1);     // initial search starting point
	RealVector xi(1);    // direction of search
	// Initialize search starting point and direction:
	p(0) = -3.;
	xi(0) = 3.;
	RealVector d = {-6};
	Testfunction function;
	LineSearch<RealVector> search;
	search.lineSearchType() = LineSearchType::Dlinmin;
	search.init(function);
	// Minimize function:
	search(p,fret, xi,d);

	// lines below are for self-testing this example, please ignore
	BOOST_CHECK_SMALL(fret-2,1.e-14);
}

BOOST_AUTO_TEST_SUITE_END()
