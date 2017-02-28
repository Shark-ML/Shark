//===========================================================================
/*!
 * \brief       Implements the Tchebycheff scalarizer.
 *
 * K. Miettinen, "Nonlinear Multiobjective Optimization", International Series
 * in Operations Research and Management Science (12)
 * DOI: 10.1007/978-1-4615-5563-6
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

#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SCALARIZERS_TCHEBYCHEFF
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SCALARIZERS_TCHEBYCHEFF

namespace shark {


double tchebycheffScalarizer(
	RealVector const & fitness,
	RealVector const & weights,
	RealVector const & optimalPointFitness
){
	auto w = weights[0] == 0 ? 1e-5 : weights[0];
	double max_fun = w * std::abs(fitness[0] - optimalPointFitness[0]);
	for(std::size_t i = 1; i < fitness.size(); ++i){
		w = weights[i] == 0 ? 1e-5: weights[i];
		max_fun = std::max(max_fun,w * std::abs(fitness[i] - optimalPointFitness[i]));
	}
	return max_fun;
}


} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SCALARIZERS_TCHEBYCHEFF
