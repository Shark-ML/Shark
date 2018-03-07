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

#include <boost/math/special_functions/binomial.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Lattice.h>

namespace shark {
namespace detail {

std::size_t sumlength(std::size_t const n, std::size_t const sum)
{
	return static_cast<std::size_t>(
		boost::math::binomial_coefficient<double>(n - 1 + sum, sum));
}

void pointLattice_helper(
	UIntMatrix & pointMatrix,
	std::size_t const rowidx,
	std::size_t const colidx,
	std::size_t const sum_rest
){
	const std::size_t n = pointMatrix.size2() - colidx;
	if(n == 1){
		pointMatrix(rowidx, colidx) = sum_rest;
	}
	else{
		std::size_t total_rows = 0;
		for(std::size_t i = 0; i <= sum_rest; ++i){
			const std::size_t submatrix_height = sumlength(n - 1, sum_rest - i);
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

} // namespace detail

std::size_t computeOptimalLatticeTicks(
	std::size_t const n,std::size_t const target_count
){
	if(n == 1){
		return target_count;
	}
	if(n == 2){
		return target_count - 1;
	}
	std::size_t dimension_ticks_count = 0;
	while(detail::sumlength(n, dimension_ticks_count) < target_count){
		++dimension_ticks_count;
	}
	return dimension_ticks_count;
}


RealMatrix weightLattice(std::size_t const n, std::size_t const sum)
{
	const std::size_t point_count = detail::sumlength(n, sum);
	UIntMatrix pointMatrix(point_count, n);
	detail::pointLattice_helper(pointMatrix, 0, 0, sum);
	RealMatrix result = pointMatrix;
	result /= static_cast<double>(sum);
	return result;
}

RealMatrix unitVectorsOnLattice(std::size_t const n, std::size_t const sum){
	RealMatrix m = weightLattice(n, sum);
	for(std::size_t i = 0; i < m.size1(); ++i){
		row(m, i) /= norm_2(row(m, i));
	}
	return m;
}

RealMatrix preferenceAdjustedUnitVectors(
	std::size_t const n,
	std::size_t const sum, 
	std::vector<Preference> const & preferences){

	const RealMatrix uv = unitVectorsOnLattice(n, sum);
	// All vectors translated for every preference plus all the centers of the
	// preferences plus the 'n' extreme endpoints (the corners):
	const std::size_t numAdjustedVectors = preferences.size() * (1 + uv.size1()) + n;
	RealMatrix adjusted(numAdjustedVectors, uv.size2());
	std::size_t row_idx = 0;
	for(auto & preference : preferences)
	{
		double r;
		RealVector v_c;
		std::tie(r, v_c) = preference;
		v_c /= norm_2(v_c);
		for(std::size_t i = 0; i < uv.size1(); ++i)
		{
			/* 
			   Equation (14) of "Evolutionary Many-objective Optimization of
			   Hybrid Electric Vehicle Control: From General Optimization to
			   Preference Articulation"
			*/
			row(adjusted, row_idx) = r * row(uv, i) + (1 - r) * v_c;
			row(adjusted, row_idx) /= norm_2(row(adjusted, row_idx));
			++row_idx;
		}
		// Put the center of the preference in the set of adjusted vectors too
		row(adjusted, row_idx) = v_c;
		++row_idx;
	}
	// Finally, we add the 'n' extreme end points of the original unit vectors
	// -- the "corners".
	for(std::size_t i = 0; i < n; ++i)
	{
		for(std::size_t j = 0; j < n; ++j)
		{
			adjusted(row_idx + i, j) = i == j ? 1 : 0;
		}
	}
	return adjusted;
}

RealMatrix preferenceAdjustedWeightVectors(
	std::size_t const n,
	std::size_t const sum,
	std::vector<Preference> const & preferences){

	RealMatrix m = preferenceAdjustedUnitVectors(n, sum, preferences);
	/* 
	   Translate vectors from the unit sphere to the hyperplane.  Equation (13)
	   of "Evolutionary Many-objective Optimization of Hybrid Electric Vehicle
	   Control: From General Optimization to Preference Articulation"
	*/
	for(std::size_t i = 0; i < m.size1(); ++i)
	{
		row(m, i) /= norm_1(row(m, i));
	}
	return m;
}

} // namespace shark
