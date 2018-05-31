#define BOOST_TEST_MODULE DirectSearch_RVEA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/RVEA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>
#include <shark/ObjectiveFunctions/Benchmarks/Hypervolumes.h>

#include "../testFunction.h"

#include <iostream>

using namespace shark;
using namespace shark::benchmarks;

typedef boost::mpl::list<DTLZ1, DTLZ2, DTLZ3, DTLZ4, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
//	DTLZ5
//	DTLZ6
//	DTLZ7
                         > obj_funs;



BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_RVEA)


BOOST_AUTO_TEST_CASE_TEMPLATE(Hypervolume_functions, OF, obj_funs)
{
	const std::size_t reps = 5;
	const std::size_t mu = 10;
	const std::size_t iters = 1000;
	const std::size_t num_objectives = 2;
	const RealVector reference(num_objectives, 11);
	OF function(5);
	const double volume = optimal_hyper_volume(function, mu);
	RVEA optimizer;
	optimizer.approxMu() = mu;
	optimizer.crossoverProbability() = 1;
	optimizer.maxIterations() = iters;
	testFunction(optimizer, function, reference, volume, reps, iters, 5.e-2);
}


BOOST_AUTO_TEST_SUITE_END()
