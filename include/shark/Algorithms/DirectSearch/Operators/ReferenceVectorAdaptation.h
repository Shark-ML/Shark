//===========================================================================
/*!
 *
 *
 * \brief		Reference vector adaptation for the RVEA algorithm.
 *
 * \author		Bjoern Bugge Grathwohl
 * \date		March 2017
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

#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H

#include <shark/LinAlg/Base.h>
#include <shark/Algorithms/DirectSearch/Individual.h>

namespace shark {
/**
 * \brief Reference vector adaptation for the RVEA algorithm.
 *
 * This operator is supposed to be applied regularly in the RVEA algorithm to
 * adjust the set of reference vectors to better match a scaled pareto front.
 * See below paper for details:
 * R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, “A reference vector guided
 * evolutionary algorithm for many-objective optimization,” IEEE Transactions on
 * Evolutionary Computation, Vol 20, Issue 5, October 2016
 * http://dx.doi.org/10.1109/TEVC.2016.2519378
 */
template <typename IndividualType>
struct ReferenceVectorAdaptation
{
	/**
	 * \brief Apply adaptation operator and update the angles.
	 */
	void operator()(
		std::vector<IndividualType> const & population,
		RealMatrix & referenceVectors,
		RealVector & minAngles)
	{
		auto f = unpenalizedFitness(population);
		const std::size_t sz = f.size();
		const std::size_t w = f[0].size();
		RealVector diff(w);
		for(std::size_t i = 0; i < w; ++i)
		{
			double max = std::numeric_limits<double>::min();
			double min = std::numeric_limits<double>::max();
			for(std::size_t j = 0; j < sz; ++j)
			{
				max = std::max(max, f[j][i]);
				min = std::min(min, f[j][i]);
			}
			double d = max - min;
			diff[i] = (d == 0) ? 1 : d;
		}
		referenceVectors = m_initVecs * repeat(diff, m_initVecs.size1());
		for(std::size_t i = 0; i < referenceVectors.size1(); ++i)
		{
			row(referenceVectors, i) /= norm_2(row(referenceVectors, i));
		}
		updateAngles(referenceVectors, minAngles);
	}

	/**
	 * \brief Compute the minimum angles between unit vectors.
	 */
	static void updateAngles(
		RealMatrix const & referenceVectors,
		RealVector & minAngles)
	{
		const std::size_t s = referenceVectors.size1();
		const RealMatrix m = acos(prod(referenceVectors,
									   trans(referenceVectors))) +
			to_diagonal(RealVector(s, 1e10));
		for(std::size_t i = 0; i < s; ++i)
		{
			minAngles[i] = min(row(m, i));
		}
	}

	template <typename Archive>
	void serialize(Archive & archive)
	{
		archive & m_initVecs;
	}

	/// \brief The set of initial reference vectors.
	///
	/// This must be set before the operator is called the first time.
	RealMatrix m_initVecs;
};

} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H
