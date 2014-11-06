#include <EALib/CMSA.h>
#include <EALib/ObjectiveFunctions.h>

#define BOOST_TEST_MODULE EALib_SchwefelEllipsoidCMSA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include<algorithm>

BOOST_AUTO_TEST_SUITE (EALib_SchwefelEllipsoidCMSA)

BOOST_AUTO_TEST_CASE( EALib_SchwefelEllipsoidCMSA )
{
	const unsigned Seed = 42;
	const unsigned Trials=30;
	const unsigned Dimension  = 10;
	const double   GlobalStepInit = 1.;

	double results[Trials];

	Rng::seed(Seed);
	SchwefelEllipsoidRotated f(Dimension);
	CMSASearch cma;

	for(size_t trial=0;trial!=Trials;++trial)
	{
		// start point
		RealVector start(Dimension);
		double* p=&start(0);
		f.ProposeStartingPoint(p);


		cma.init(f, start, GlobalStepInit);

		// optimization loop
		size_t i = 0;
		do {
			cma.run();
			i++;
		} while (cma.bestSolutionFitness() > 10E-10);
		results[trial]=i;
	}
	//sort and check the median
	std::sort(results,results+Trials);
	//median should lie between 250 and 278
	BOOST_CHECK_CLOSE(results[Trials/2],265,5);
}

BOOST_AUTO_TEST_SUITE_END()
