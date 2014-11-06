#include <EALib/CMSA.h>
#include <EALib/ObjectiveFunctions.h>

#define BOOST_TEST_MODULE EALib_SchwefelEllipsoidCMSA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include<algorithm>

BOOST_AUTO_TEST_SUITE (EALib_paraboloidCMSA)

BOOST_AUTO_TEST_CASE( EALib_ParaboloidCMA )
{
	const unsigned Seed = 44;
	const unsigned Trials=30;
	const unsigned Dimension = 8;

	double results[Trials];

	Rng::seed(Seed);

	//
	// fitness function
	//
	const unsigned a = 1000;  // determines (square root of) problem condition
	Paraboloid f(Dimension, a);

	//
	// EA parameters
	//
	const unsigned Iterations     = 1000;
	const double   MinInit        = .1;
	const double   MaxInit        = .3;
	const double   GlobalStepInit = 1.;


	CMSASearch cma;

	for(size_t trial=0;trial!=Trials;++trial)
	{

		// start point
		RealVector start=blas::scalar_vector<double>(Dimension,Rng::uni(MinInit, MaxInit));
		cma.init(f, start, GlobalStepInit);

		for (size_t i=0; i<Iterations; i++)
		{
			cma.run();
		}
		results[trial]=cma.bestSolutionFitness();
	}
	//sort results and test the median
	std::sort(results,results+Trials);
	BOOST_CHECK_SMALL(results[Trials/2],1.e-14);
}

BOOST_AUTO_TEST_SUITE_END()
