#define BOOST_TEST_MODULE DirectSearch_Operators_Grid

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Grid.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

#include <iostream>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_Operators_Grid)

BOOST_AUTO_TEST_CASE(sumsto_correct)
{
    for(std::size_t n = 2; n < 6; ++n)
    {
        for(std::size_t sum = 3; sum < 10; ++sum)
        {
            UIntMatrix m = sumsto(n, sum);
            for(std::size_t row = 0; row < m.size1(); ++row)
            {
                std::size_t actual_sum = 0;
                for(std::size_t col = 0; col < m.size2(); ++col)
                {
                    actual_sum += m(row, col);
                }
                BOOST_CHECK_EQUAL(actual_sum, sum);
            }   
        }
    }
}

BOOST_AUTO_TEST_CASE(sumsto_rec_correct)
{
    const std::size_t sum = 10;
    const std::size_t n_max = 10;
    for(std::size_t n = 2; n < n_max; ++n)
    {
        std::list<std::list<std::size_t>> ls = sumsto_rec(n, sum);
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


BOOST_AUTO_TEST_CASE(best_point_count_2d_correct)
{
    for(std::size_t i = 1; i < 100; ++i)
    {
        const std::size_t pc = bestPointCountForLattice(2, i);
        BOOST_CHECK_EQUAL(i, sumlength(2, pc));
    }
}


BOOST_AUTO_TEST_CASE(sumsto_has_expected_size)
{
    for(std::size_t mu_prime = 3; mu_prime < 10; ++mu_prime)
    {
        for(std::size_t d = 2; d < 5; ++d)
        {
            std::list<std::list<std::size_t>> ls = sumsto_rec(d, mu_prime);
            std::size_t expected_size = sumlength(d, mu_prime);
            BOOST_CHECK_EQUAL(expected_size, ls.size());
        }
    }
}

BOOST_AUTO_TEST_CASE(weights_expected_size)
{
    for(std::size_t n = 2; n < 6; ++n)
    {
        for(std::size_t sum = 3; sum < 15; ++sum)
        {
            std::size_t exp = sumlength(n, sum);
            RealMatrix w = weightLattice(n, sum);
            BOOST_CHECK_EQUAL(w.size1(), exp);
        }
    }
}

BOOST_AUTO_TEST_CASE(best_point_count)
{
    for(std::size_t n = 2; n < 5; ++n)
    {
        for(std::size_t sum = 3; sum < 10; ++sum)
        {
            std::size_t b = bestPointCountForLattice(n, sum);
            RealMatrix w = weightLattice(n, b);
            BOOST_CHECK(sumlength(n, b) >= sum);
            BOOST_CHECK_EQUAL(w.size1(), sumlength(n, b));
        }
    }
}


BOOST_AUTO_TEST_CASE(lattice_sum_to_one)
{
    const std::size_t mu_prime = 12;
    const std::size_t n_max = 10;
    for(std::size_t n = 2; n < n_max; ++n)
    {
        const RealMatrix weights = weightLattice(n, mu_prime);
        for(std::size_t row = 0; row < weights.size1(); ++row)
        {
            double sum = 0;
            std::for_each(weights.row_begin(row), weights.row_end(row),
                          [&sum](double d)
                          {
                              sum += d;
                          });
            BOOST_CHECK_CLOSE(1.0, sum, 1e-8);
        }
    }
}

BOOST_AUTO_TEST_CASE(vector_sorting_correct)
{
    const std::size_t mu_prime = 8;
    const std::size_t n_max = 4;
    for(std::size_t n = 2; n < n_max; ++n)
    {
        const RealMatrix weights = weightLattice(mu_prime, n);
        for(std::size_t T = 1; T <= weights.size1() / 2 && T <= 30; ++T)
        {
            UIntMatrix dists = closestIndices(weights, T);
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
                    // delta is negative if the vector is *not* sorted
                    if(delta < 0)
                    {
                        // Check whether it's just due to floating point stuff.
                        // (This is why we don't just use std::is_sorted)
                        BOOST_CHECK_CLOSE_FRACTION(my_dists_vec[i], 
                                                   my_dists_vec[i - 1], 1e-8);
                    }
                }
            }
        }
        
    }
}


BOOST_AUTO_TEST_SUITE_END()
