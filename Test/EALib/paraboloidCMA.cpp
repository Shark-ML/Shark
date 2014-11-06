#include <EALib/CMA.h>
#include <EALib/ObjectiveFunctions.h>

#define BOOST_TEST_MODULE EALib_ParaboloidCMA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <algorithm>
#include <iostream>

BOOST_AUTO_TEST_SUITE (EALib_paraboloidCMA)

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
	const unsigned Iterations     = 650;
	const double   MinInit        = .1;
	const double   MaxInit        = .3;
	const double   GlobalStepInit = 1.;


	CMASearch cma;

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
		std::cout<<trial<<" "<<cma.bestSolutionFitness()<<std::endl;
	}
	//sort results and test the median
	std::sort(results,results+Trials);
	BOOST_CHECK_SMALL(results[Trials/2],1.e-14);
}

BOOST_AUTO_TEST_SUITE_END()
