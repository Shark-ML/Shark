#define BOOST_TEST_MODULE DirectSearch_MOEAD
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/MOEAD.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

#include "../testFunction.h"

#include <iostream>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_MOEAD)


BOOST_AUTO_TEST_CASE(HYPERVOLUME_Functions)
{
    const std::size_t reps = 10;
    const std::size_t mu = 100;
    const std::size_t T = 10;
    const std::size_t iters = 500 * mu;
    const RealVector reference{11, 11};
    {
        DTLZ1 function(5);
//      const double volume = 120.861111; // mu = 10
        const double volume = 120.873737; // mu = 100
        MOEAD optimizer;
        optimizer.mu() = mu;
        optimizer.neighbourhoodSize() = T;
        testFunction(optimizer, function, reference, volume, reps, iters, 5.e-3);
    }
    {
        DTLZ2 function(5);
//      const double volume = 120.178966; // mu = 10
        const double volume = 120.210644; // mu = 100
        MOEAD optimizer;
        optimizer.mu() = mu;
        optimizer.neighbourhoodSize() = T;
        testFunction(optimizer, function, reference, volume, reps, iters, 5.e-3);
    }
    {
        ZDT1 function(5);
//		const double volume = 120.613761; // mu = 10
        const double volume = 120.662137; // mu = 100
        MOEAD optimizer;
        optimizer.mu() = mu;
        optimizer.neighbourhoodSize() = T;
        testFunction(optimizer, function, reference, volume, reps, iters, 5.e-3);
    }
	{
		ZDT2 function(5);
//		const double volume = 120.286820; // mu = 10
        const double volume = 120.328881; // mu = 100
        MOEAD optimizer;
		optimizer.mu() = mu;
        optimizer.neighbourhoodSize() = T;
		testFunction(optimizer, function, reference, volume, reps, iters, 5.e-3);
	}
	{
		ZDT3 function(5);
//		const double volume = 128.748470; // mu = 10
        const double volume = 128.775955; // mu = 100
		MOEAD optimizer;
		optimizer.mu() = mu;
        optimizer.neighbourhoodSize() = T;
		testFunction(optimizer, function, reference, volume, reps, iters, 5.e-3);
	}
    {
        ZDT4 function(5);
//      const double volume = 120.613761; // mu = 10
        const double volume = 120.662137; // mu = 100
        MOEAD optimizer;
        optimizer.mu() = mu;
        optimizer.neighbourhoodSize() = T;
        testFunction(optimizer, function, reference, volume, reps, iters, 5.e-3);
    }
    {
        ZDT6 function(5);
//      const double volume = 117.483246; // mu = 10
        const double volume = 117.514950; // mu = 100
        MOEAD optimizer;
        optimizer.mu() = mu;
        optimizer.neighbourhoodSize() = T;
        testFunction(optimizer, function, reference, volume, reps, iters, 5.e-3);
    }
}




BOOST_AUTO_TEST_SUITE_END()
