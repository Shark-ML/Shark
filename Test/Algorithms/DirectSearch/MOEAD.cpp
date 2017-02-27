#define BOOST_TEST_MODULE DirectSearch_MOEAD
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/MOEAD.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

#include "../testFunction.h"

#include <iostream>

using namespace shark;

typedef boost::mpl::list<DTLZ1, DTLZ2, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6> obj_funs;

double optimal_hyper_volume(std::size_t n, const DTLZ1 &) {
    if(n == 10) return 120.86111;
    if(n == 100) return 120.873737; 
    return -1;
}
double optimal_hyper_volume(std::size_t n, const DTLZ2 &) { 
    if(n == 10) return 120.178966;
    if(n == 100) return 120.210644; 
    return -1;
}
double optimal_hyper_volume(std::size_t n, const ZDT1 &) { 
    if(n == 10) return 120.613761;
    if(n == 100) return 120.662137; 
    return -1;
}
double optimal_hyper_volume(std::size_t n, const ZDT2 &) { 
    if(n == 10) return 120.286820;
    if(n == 100) return 120.328881; 
    return -1;
}
double optimal_hyper_volume(std::size_t n, const ZDT3 &) { 
    if(n == 10) return 128.748470;
    if(n == 100) return 128.775955; 
    return -1;
}
double optimal_hyper_volume(std::size_t n, const ZDT4 &) { 
    if(n == 10) return 120.613761;
    if(n == 100) return 120.662137; 
    return -1;
}
double optimal_hyper_volume(std::size_t n, const ZDT6 &) { 
    if(n == 10) return 117.483246;
    if(n == 100) return 117.514950; 
    return -1;
}


BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_MOEAD)


BOOST_AUTO_TEST_CASE_TEMPLATE(Hypervolume_functions_mu10, OF, obj_funs)
{
    const std::size_t reps = 5;
    const std::size_t mu = 10;
    const std::size_t T = 5;
    const std::size_t iters = 1000 * mu;
    const RealVector reference{11, 11};
    OF function(5);
    const double volume = optimal_hyper_volume(mu, function);
    MOEAD optimizer;
    optimizer.mu() = mu;
    optimizer.neighbourhoodSize() = T;
    testFunction(optimizer, function, reference, volume, reps, iters, 5.e-2);
}



BOOST_AUTO_TEST_SUITE_END()
