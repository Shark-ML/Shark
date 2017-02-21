//===========================================================================
/*!
 * 
 *
 * \brief       Various functions for generating n-dimensional grids.
 * 
 *
 * \author      Bjørn Bugge Grathwohl
 * \date        February 2017
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================

#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_GRID
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_GRID

#include <boost/math/special_functions/binomial.hpp>

#include <shark/LinAlg/Base.h>
#include <shark/LinAlg/Metrics.h>

namespace shark {


// The number of n-points that sum up to 'sum'
std::size_t sumlength(std::size_t const n, std::size_t const sum)
{
    const std::size_t d = n - 2;
    return static_cast<std::size_t>(
        boost::math::binomial_coefficient<double>(d + sum + 1, sum));
}

// A list of n-dimensional points that each sums to "sum".
std::list<std::list<std::size_t>> sumsto(std::size_t const n, 
                                         std::size_t const sum)
{
    SIZE_CHECK(n > 1);
    if(n == 2)
    {
        std::list<std::list<std::size_t>> vs;
        for(std::size_t i = 0; i <= sum; ++i)
        {
            vs.push_back(std::list<std::size_t>{i, sum - i});
        }
        return vs;
    }
    else // n > 2
    {
        std::list<std::list<std::size_t>> vs;
        for(std::size_t i = 0; i <= sum; ++i)
        {
            for(auto & v_sub : sumsto(n - 1, sum - i))
            {
                v_sub.push_front(i);
                vs.push_back(v_sub);
            }
        }
        return vs;
    }
}

// Gives the number of ticks in each dimension required to make an n-dimensional
// lattice where the total number of points is as close to 'target_count' as possible:
// Given n and target_count returns T such that
//     forall T'<T . sumlength(n,T') < target_count
// and forall T'>T . sumlength(n,T') > target_count
std::size_t bestPointCountForLattice(std::size_t const n, 
                                     std::size_t const target_count)
{
    SIZE_CHECK(n > 1);
    std::size_t cur = 0;
    std::size_t dimension_ticks_count = 0;
    const std::size_t d = n - 2;
    while(cur < target_count)
    {
        cur += static_cast<std::size_t>(
            boost::math::binomial_coefficient<double>(
                dimension_ticks_count + d, d));
        ++dimension_ticks_count;
    }
    return dimension_ticks_count;
}

UIntMatrix pointLattice(std::size_t const n, std::size_t const sum)
{
    typedef std::list<std::size_t> point_t;
    std::list<point_t> points = sumsto(n, sum);
    UIntMatrix point_matrix(points.size(), n);
    std::size_t row = 0;
    for(point_t & point : points)
    {
        std::copy(point.begin(), point.end(), point_matrix.row_begin(row));
        ++row;
    }
    return point_matrix;
}

RealMatrix weightLattice(std::size_t const n, 
                         std::size_t const sum)
{
    return static_cast<RealMatrix>(pointLattice(n, sum)) / sum;
}

// For each row in the matrix, give the indices of the 't' closest vectors
// sorted ascendingly.  returns a m.size(1) * t matrix with the indices.
UIntMatrix closestIndices(RealMatrix const & m, 
                          std::size_t const t)
{
    const RealMatrix distances = remora::distanceSqr(m, m);
    UIntMatrix neighbourIndices(m.size1(), t);
    // For each vector we are interested in indices of the t closest vectors.
    for(std::size_t i = 0; i < m.size1(); ++i)
    {
        const RealVector my_dists(distances.row_begin(i),
                                  distances.row_end(i));
        // Make some indices we can sort.
        std::vector<std::size_t> indices(my_dists.size());
        std::iota(indices.begin(), indices.end(), 0);
        // Sort indices by the distances.
        std::sort(indices.begin(), indices.end(),
                  [&](std::size_t a, std::size_t b)
                  {
                      return my_dists[a] < my_dists[b];
                  });
        // Copy the T closest indices into B.
        std::copy_n(indices.begin(), t, neighbourIndices.row_begin(i));
    }
    return neighbourIndices;
}

void dump(RealMatrix const & m, std::string const & filename)
{
    std::ofstream file(filename);
    for(std::size_t row = 0; row < m.size1(); ++row)
    {
        std::for_each(m.row_begin(row), m.row_end(row),
                      [&file](double x)
                      {
                          file << x << "\t";
                      });
        file << std::endl;
    }
}


} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_GRID
