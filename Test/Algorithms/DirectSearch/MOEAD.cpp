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
    const std::size_t mu = 100;
    const std::size_t T = 10;
    const std::size_t iters = 1000;
    const RealVector reference{11, 11};
    {
        DTLZ1 function(5);
//        const double volume = 120.861111; // mu = 10
        const double volume = 120.873737; // mu = 100
        MOEAD optimizer;
        optimizer.mu() = mu;
        optimizer.neighbourhoodSize() = T;
        testFunction(optimizer, function, reference, volume, 10, iters, 5.e-3);
    }
    {
        DTLZ2 function(5);
//        const double volume = 120.178966; // mu =10
        const double volume = 120.210644; // mu =100
        MOEAD optimizer;
        optimizer.mu() = mu;
        optimizer.neighbourhoodSize() = T;
        testFunction(optimizer, function, reference, volume, 10, iters, 5.e-3);
    }
    {
        ZDT1 function(5);
//		const double volume = 120.613761; // mu = 10
        const double volume = 120.662137; // mu = 100
        MOEAD optimizer;
        optimizer.mu() = mu;
        optimizer.neighbourhoodSize() = T;
        testFunction(optimizer, function, reference, volume, 10, iters, 5.e-3);
    }
	{
		ZDT2 function(5);
//		const double volume = 120.286820; // mu = 10
        const double volume = 120.328881; // mu = 100
        MOEAD optimizer;
		optimizer.mu() = mu;
        optimizer.neighbourhoodSize() = T;
		testFunction(optimizer, function, reference, volume, 10, iters, 5.e-3);
	}
	{
		ZDT3 function(5);
//		const double volume = 128.748470; // mu = 10
        const double volume = 128.775955; // mu = 100
		MOEAD optimizer;
		optimizer.mu() = mu;
        optimizer.neighbourhoodSize() = T;
		testFunction(optimizer, function, reference, volume, 10, iters, 5.e-3);
	}
	{
		ZDT6 function(5);
//		const double volume = 117.483246; // mu = 10
        const double volume = 117.514950; // mu = 100
		MOEAD optimizer;
		optimizer.mu() = mu;
		optimizer.neighbourhoodSize() = T;
		testFunction(optimizer, function, reference, volume, 10, iters, 5.e-3);
	}
}


BOOST_AUTO_TEST_CASE(sumsto_correct)
{
    const std::size_t sum = 10;
    const std::size_t n_max = 10;
    for(std::size_t n = 2; n < n_max; ++n)
    {
        std::list<std::list<std::size_t>> ls = detail::sumsto(n, sum);
        for(std::list<std::size_t> & l : ls)
        {
            std::size_t actual_sum = 0;
            for(auto x : l)
            {
                actual_sum += x;
            }
            BOOST_CHECK_EQUAL(actual_sum, sum);
        }
    }
}


BOOST_AUTO_TEST_CASE(sumsto_correct_2)
{
    for(std::size_t mu_prime = 3; mu_prime < 10; ++mu_prime)
    {
        for(std::size_t d = 2; d < 5; ++d)
        {
            std::list<std::list<std::size_t>> ls = detail::sumsto(d, mu_prime);
            std::size_t expected_size = 0;
            for(std::size_t i = 0; i <= mu_prime; ++i)
            {
                expected_size += n_choose_k(i + d - 2, d - 2);
            }
            BOOST_CHECK_EQUAL(expected_size, ls.size());
        }
    }
}

BOOST_AUTO_TEST_CASE(lattice_correct)
{
    const std::size_t mu_prime = 12;
    const std::size_t n_max = 10;
    for(std::size_t n = 2; n < n_max; ++n)
    {
        const RealMatrix weights = detail::uniformWeightVectorLattice(mu_prime, n);
        for(std::size_t row = 0; row < weights.size1(); ++row)
        {
            double sum = 0;
            std::for_each(weights.row_begin(row), weights.row_end(row),
                          [&sum](double d)
                          {
                              sum += d;
                          });
            BOOST_CHECK_CLOSE(1.0, sum, 0.000001);
        }
    }
}

BOOST_AUTO_TEST_CASE(weight_vector_sorting_correct)
{
    const std::size_t mu_prime = 8;
    const std::size_t n_max = 4;
    for(std::size_t n = 2; n < n_max; ++n)
    {
        const RealMatrix weights = detail::uniformWeightVectorLattice(mu_prime, n);
        for(std::size_t T = 1; T <= weights.size1() / 2 && T <= 30; ++T)
        {
            UIntMatrix dists = detail::getClosestWeightVectors(weights, T);
            for(std::size_t row = 0; row < dists.size1(); ++row)
            {
                std::list<std::vector<double>> my_nearest_points;
                std::for_each(dists.row_begin(row), dists.row_end(row),
                              [&](std::size_t idx)
                              {
                                  my_nearest_points.push_back(
                                      std::vector<double>(weights.row_begin(idx),
                                                          weights.row_end(idx)));
                              });
                BOOST_CHECK_EQUAL(my_nearest_points.size(), T);
                const std::vector<double> this_point(weights.row_begin(row),
                                                     weights.row_end(row));
                std::list<double> my_dists;
                for(std::vector<double> const & point : my_nearest_points)
                {
                    double d = 0;
                    for(std::size_t i = 0; i < point.size(); ++i)
                    {
                        d += std::pow(this_point[i] - point[i], 2);
                    }
                    my_dists.push_back(std::sqrt(d));
                }
                BOOST_CHECK_EQUAL(my_dists.size(), T);
                std::vector<double> my_dists_vec(my_dists.begin(), my_dists.end());
                for(std::size_t i = 1; i < my_dists_vec.size(); ++i)
                {
                    double delta = my_dists_vec[i] - my_dists_vec[i - 1];
                    if(delta < 0)
                    {
                        BOOST_CHECK_CLOSE_FRACTION(my_dists_vec[i], 
                                                   my_dists_vec[i - 1], 1e-8);
                    }
                }
                // We cannot just use std::is_sorted because the distances are
                // very close to one another due to floating point instability
                // the order is not 100% guaranteed.  Instead we check a relaxed
                // version, above, which just asserts that whenever elements are
                // not in sorted order it is because they are "close enough" to
                // one another to be essentially identical.
//                BOOST_CHECK(std::is_sorted(my_dists.begin(), my_dists.end()));
            }
        }
        
    }
}


BOOST_AUTO_TEST_SUITE_END()
