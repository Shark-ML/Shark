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
#include <shark/Rng/DiscreteUniform.h>

namespace shark {


/*
 * An n-dimensional point sums to 's' if the sum of the parts equal 's',
 * e.g. the point (x_0,x_1,x_2) sums to x_0+x_1+x+2 etc.  The number of
 * n-dimensional points that sum to 's' is given by the formula "N over K" where
 * N is n - 2 + s + 1 and K is s. 
 */
std::size_t sumlength(std::size_t const n, std::size_t const sum)
{
    return static_cast<std::size_t>(
        boost::math::binomial_coefficient<double>(n - 1 + sum, sum));
}

std::list<std::list<std::size_t>> sumsto_rec(std::size_t const n, 
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
            for(auto & v_sub : sumsto_rec(n - 1, sum - i))
            {
                v_sub.push_front(i);
                vs.push_back(v_sub);
            }
        }
        return vs;
    }
}

namespace detail {

void pointLattice_helper(UIntMatrix & pointMatrix,
                         std::size_t const rowidx,
                         std::size_t const colidx,
                         std::size_t const sum_rest)
{
    const std::size_t n = pointMatrix.size2() - colidx;
    if(n == 1)
    {
        pointMatrix(rowidx, colidx) = sum_rest;
    }
    else
    {
        std::size_t total_rows = 0;
        for(std::size_t i = 0; i <= sum_rest; ++i)
        {
            const std::size_t submatrix_height = sumlength(n - 1, 
                                                           sum_rest - i);
            // Each first entry in submatrix contains i, and remaining columns
            // in each row all sum to sum_rest - i.
            for(std::size_t j = 0; j < submatrix_height; ++j)
            {
                pointMatrix(total_rows + rowidx + j, colidx) = i;
            }
            pointLattice_helper(pointMatrix, total_rows + rowidx,
                                colidx + 1, sum_rest - i);
            total_rows += submatrix_height;
        }
    }
}

// A corner is a point where exactly one dimension is non-zero.
template <typename Iterator> 
bool isCorner(Iterator begin, Iterator end)
{
    std::size_t nonzero = 0;
    for(auto iter = begin; iter != end; ++iter)
    {
        if(nonzero > 1)
        {
            return false;
        }
        if(*iter > 0)
        {
            ++nonzero;
        }
    }
    return nonzero == 1;
}

} // namespace detail

/*
 * Generates a matrix where each row is an n-dimensional point that sums to
 * 'sum'. These points are all on the (n-1)-dimensional simplex, i.e., when the
 * points are 3-dimensional the points are on a triangle, when the points are
 * 2-dimensional they are on a line etc.  An (n-1)-dimensional simplex has n
 * corners which are the points where exactly one dimension is non-zero.
 */
UIntMatrix pointLattice(std::size_t const n, std::size_t const sum)
{
    const std::size_t point_count = sumlength(n, sum);
    UIntMatrix pointMatrix(point_count, n);
    detail::pointLattice_helper(pointMatrix, 0, 0, sum);
    return pointMatrix;
}


/*
 * Sample points uniformly from the simplex given in the matrix.  Corners are
 * always included in the sampled point set (unless excplicitly turned off with
 * keep_corners set to false).  The returned matrix will always have n points or
 * the same number of points as the original matrix if n is smaller.  
 */
template <typename Matrix, typename RngType = shark::DefaultRngType>
Matrix sampleUniformly(RngType & rng, Matrix const & matrix, 
                       std::size_t const n,
                       bool const keep_corners = true)
{
    // No need to do all the below stuff if we're gonna grab it all anyway.
    if(matrix.size1() <= n) 
    {
        return matrix;
    }
    Matrix sampledMatrix(n, matrix.size2());
    std::set<std::size_t> added_rows;
    // First find all the corners and add them to our set of sampled points.
    if(keep_corners)
    {
        for(std::size_t row = 0; row < matrix.size1(); ++row)
        {
            if(detail::isCorner(matrix.row_begin(row), matrix.row_end(row)))
            {
                added_rows.insert(row);
            }
        }
    }
    DiscreteUniform<RngType> uni(rng, 0, matrix.size1() - 1);
    while(added_rows.size() < n)
    {
        // If we sample an existing index it doesn't alter the set.
        added_rows.insert(uni());
    }
    std::size_t i = 0;
    for(std::size_t row_idx : added_rows)
    {
        std::copy(matrix.row_begin(row_idx), matrix.row_end(row_idx),
                  sampledMatrix.row_begin(i));
        ++i;
    }
    return sampledMatrix;
}

// Gives the number of ticks in each dimension required to make an n-dimensional
// lattice where the total number of points is as close to 'target_count' as possible:
// Given n and target_count returns T such that
//     forall T'<T . sumlength(n,T') < target_count
// and forall T'>T . sumlength(n,T') > target_count
std::size_t bestPointSumForLattice(std::size_t const n, 
                                   std::size_t const target_count)
{
    if(n == 1)
    {
        return target_count;
    }
    if(n == 2)
    {
        return target_count - 1;
    }
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
