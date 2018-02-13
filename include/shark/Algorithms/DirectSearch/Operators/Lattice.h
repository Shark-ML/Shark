//===========================================================================
/*!
 *
 * \brief   Various functions for generating n-dimensional grids
 *          (simplex-lattices).
 *
 *
 * \author   Bjørn Bugge Grathwohl
 * \date     February 2017
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================

#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_LATTICE
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_LATTICE

#include <shark/LinAlg/Base.h>
#include <shark/LinAlg/Metrics.h>
#include <shark/Core/Random.h>

#include <numeric>

namespace shark {

namespace detail {

/*
 * An n-dimensional point sums to 's' if the sum of the parts equal 's',
 * e.g. the point (x_0,x_1,x_2) sums to x_0+x_1+x+2 etc.  The number of
 * n-dimensional points that sum to 's' is given by the formula "N over K" where
 * N is n - 2 + s + 1 and K is s.
 */
std::size_t sumlength(std::size_t const n, std::size_t const sum);

void pointLattice_helper(
	UIntMatrix & pointMatrix,
	std::size_t const rowidx,
	std::size_t const colidx,
	std::size_t const sum_rest);

/// \brief A corner is a point where exactly one dimension is non-zero.
template <typename Iterator>
bool isLatticeCorner(Iterator begin, Iterator end){
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
 * Sample points uniformly and uniquely from the simplex given in the matrix.
 * Corners are always included in the sampled point set (unless excplicitly
 * turned off with keep_corners set to false). The returned matrix will always
 * have n points or the same number of points as the original matrix if n is
 * smaller.
 */
template <typename Matrix, typename randomType = shark::random::rng_type>
Matrix sampleLatticeUniformly(
	randomType & rng, Matrix const & matrix,
	std::size_t const n,
	bool const keep_corners = true
){
	// No need to do all the below stuff if we're gonna grab it all anyway.
	if(matrix.size1() <= n){
		return matrix;
	}
	Matrix sampledMatrix(n, matrix.size2());
	std::set<std::size_t> added_rows;
	// First find all the corners and add them to our set of sampled points.
	if(keep_corners){
		for(std::size_t row = 0; row < matrix.size1(); ++row){
			if(detail::isLatticeCorner(matrix.major_begin(row), matrix.major_end(row))){
				added_rows.insert(row);
			}
		}
	}
	while(added_rows.size() < n){
		// If we sample an existing index it doesn't alter the set.
		added_rows.insert(random::discrete(rng, std::size_t(0), matrix.size1() - 1));
	}
	std::size_t i = 0;
	for(std::size_t major_idx : added_rows)
	{
		std::copy(
			matrix.major_begin(major_idx), matrix.major_end(major_idx),
			sampledMatrix.major_begin(i)
		);
		++i;
	}
	return sampledMatrix;
}

/// \brief A preferred region in a lattice-sampled unit sphere.
///
/// A preferred region is a pair with a radius and a vector.  The intersection
/// of the vector and the unit sphere denotes the center of the preferred
/// region, and the radius denotes the radius of a sphere constructed such that
/// the center point (intersection of vector and unit sphere) is on its
/// periphery.  Points are then sampled on this smaller sphere and projected up
/// on the unit sphere, thus restricting the covered
/// segment/area/volume/hypervolume of the unit sphere.  See Figure 1 in
/// "Evolutionary Many-objective Optimization of Hybrid Electric Vehicle
/// Control: From General Optimization to Preference Articulation"
typedef std::pair<double, RealVector> Preference;

/// \brief Return a set of evenly spaced n-dimensional points on the unit sphere
/// clustered around the specified preference points.
RealMatrix preferenceAdjustedUnitVectors(
	std::size_t const n, 
	std::size_t const sum,
	std::vector<Preference> const & preferences);

/// \brief Return a set of of evenly spaced n-dimensional points on the "unit
/// simplex" clustered around the specified preference points.
RealMatrix preferenceAdjustedWeightVectors(
	std::size_t const n,
	std::size_t const sum,
	std::vector<Preference> const & preferences);

///\brief  Computes the number of Ticks for a grid of a certain size
///
///
/// Computes the least number of ticks in each dimension required to make an
/// n-dimensional simplex grid at least as many points as a target number points.	
/// For example, the points in a two-dimensional
/// grid -- a line -- with size n are the points (0,n-1), (1,n-2), ... (n-1,0).
std::size_t computeOptimalLatticeTicks(
	std::size_t const n, std::size_t const target_count
);

/// \brief Returns a set of evenly spaced n-dimensional points on the "unit simplex".
RealMatrix weightLattice(std::size_t const n, std::size_t const sum);

/// \brief Return a set of evenly spaced n-dimensional points on the unit sphere.
RealMatrix unitVectorsOnLattice(std::size_t const n, std::size_t const sum);

/*
 * Computes the pairwise euclidean distance between all row vectors in the
 * matrix and returns a matrix containing, for each row vector, the indices of
 * the 'n' closest row vectors.
 */
template <typename Matrix>
UIntMatrix computeClosestNeighbourIndicesOnLattice(
	Matrix const & m, std::size_t const n){
	const RealMatrix distances = distanceSqr(m, m);
	UIntMatrix neighbourIndices(m.size1(), n);
	// For each vector we are interested in indices of the t closest vectors.
	for(std::size_t i = 0; i < m.size1(); ++i)
	{
		// Make some indices we can sort.
		std::vector<std::size_t> indices(distances.size2());
		std::iota(indices.begin(), indices.end(), 0);
		// Sort indices by the distances.
		std::sort(indices.begin(), indices.end(),
		          [&](std::size_t a, std::size_t b)
		          {
			          return distances(i, a) < distances(i, b);
		          });
		// Copy the T closest indices into B.
		std::copy_n(indices.begin(), n, neighbourIndices.major_begin(i));
	}
	return neighbourIndices;
}


} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_LATTICE
