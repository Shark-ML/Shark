#define BOOST_TEST_MODULE DirectSearch_MOEAD
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/MOEAD.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>
#include <shark/ObjectiveFunctions/Benchmarks/Hypervolumes.h>

#include "../testFunction.h"

#include <iostream>

using namespace shark;
using namespace shark::benchmarks;

typedef boost::mpl::list<DTLZ1, DTLZ2, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6> obj_funs;


BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_MOEAD)


BOOST_AUTO_TEST_CASE_TEMPLATE(Hypervolume_functions_mu10, OF, obj_funs)
{
	const std::size_t reps = 5;
	const std::size_t mu = 10;
	const std::size_t T = 5;
	const std::size_t iters = 1000 * mu;
	const RealVector reference{11, 11};
	OF function(5);
	const double volume = optimal_hyper_volume(function, mu);
	MOEAD optimizer;
	optimizer.mu() = mu;
	optimizer.neighbourhoodSize() = T;
	testFunction(optimizer, function, reference, volume, reps, iters, 5.e-2);
}



BOOST_AUTO_TEST_SUITE_END()
