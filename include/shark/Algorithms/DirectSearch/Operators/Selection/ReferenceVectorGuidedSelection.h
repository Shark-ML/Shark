//===========================================================================
/*!
 *
 *
 * \brief		Implements the reference vector selection for RVEA
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

#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_REFERENCE_VECTOR_GUIDED_SELECTION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_REFERENCE_VECTOR_GUIDED_SELECTION_H

#include <shark/LinAlg/Base.h>
#include <shark/Algorithms/DirectSearch/Individual.h>

namespace shark {

/**
 * \brief Implements the reference vector selection for the RVEA algorithm.
 *
 * This selector uses a set of unit reference vectors to partition the search
 * space by assigning to each reference vector the individual that is "closest"
 * to it, as measured by the angle-penalized distance.
 * See below paper for details:
 * R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, “A reference vector guided
 * evolutionary algorithm for many-objective optimization,” IEEE Transactions on
 * Evolutionary Computation, Vol 20, Issue 5, October 2016
 * http://dx.doi.org/10.1109/TEVC.2016.2519378
 */
template <typename IndividualType>
struct ReferenceVectorGuidedSelection
{
	typedef std::set<std::size_t> bag_t;

	/**
	 * \brief Select individuals by marking them as "selected".
	 *
	 * The selection operator requires the set of reference vectors, the set of
	 * least angles between reference vectors (the gammas), as well as the
	 * current iteration number.
	 */
	void operator()(
		std::vector<IndividualType> & population,
		RealMatrix const & referenceVectors,
		RealVector const & gammas,
		std::size_t const curIteration)
	{
		if(population.empty())
		{
			return;
		}
		RealMatrix fitness = extractPopulationFitness(population);
		SIZE_CHECK(fitness.size2() == referenceVectors.size2());
		const std::size_t groupCount = referenceVectors.size1();
		// Objective value translation
		// line 4
		const RealVector minFitness = minCol(fitness);
		// line 5-7
		fitness -= repeat(minFitness, fitness.size1());

		// Population partition
		// line 9-13
		const RealMatrix cosA = cosAngles(fitness, referenceVectors);
		const RealMatrix angles = acos(cosA);
		// line 14-17
		const std::vector<bag_t> subGroups = populationPartition(cosA);
		SIZE_CHECK(subGroups.size() == groupCount);
		// Elitism selection
		for(auto & p : population)
		{
			p.selected() = false;
		}
		const double theta = fitness.size2()
			* std::pow(static_cast<double>(curIteration) /
					   static_cast<double>(m_maxIters), m_alpha);
		// line 25-28
		for(std::size_t j = 0; j < groupCount; ++j)
		{
			if(subGroups[j].size() == 0)
			{
				continue;
			}
			std::size_t selected_idx = 0;
			double min = 1e5;
			for(std::size_t i : subGroups[j])
			{
				// Angle-penalized distance (APD) calculation
				double apd = 1 + theta * angles(i, j) / gammas[j];
				apd *= norm_2(row(fitness, i));
				if(apd < min)
				{
					selected_idx = i;
					min = apd;
				}
			}
			population[selected_idx].selected() = true;
		}
	}

	/**
	 * \brief Associates a population to a set of reference vectors.
	 *
	 * The parameter is an N-by-M matrix where N is the population size and M is
	 * the number of reference vectors.  Entry (i,j) is the cosine of the angle
	 * between population i and reference vector j.
	 */
	static std::vector<bag_t> populationPartition(
		RealMatrix const & cosAngles)
	{
		std::vector<std::set<std::size_t>> subGroups(cosAngles.size2());
		for(std::size_t i = 0; i < cosAngles.size1(); ++i)
		{
			const std::size_t k = std::distance(
				row(cosAngles, i).begin(),
				std::max_element(row(cosAngles, i).begin(),
								 row(cosAngles, i).end()));
			subGroups[k].insert(i);
		}
		return subGroups;
	}

	static RealMatrix extractPopulationFitness(
		std::vector<IndividualType> const & population)
	{
		RealMatrix fitness(population.size(),
						   population[0].unpenalizedFitness().size());
		for(std::size_t i = 0; i < population.size(); ++i)
		{
			row(fitness, i) = population[i].unpenalizedFitness();
		}
		return fitness;
	}

	static RealVector minCol(RealMatrix const & m)
	{
		RealVector minColumns(m.size2());
		for(std::size_t i = 0; i < m.size2(); ++i)
		{
			minColumns[i] = min(column(m, i));
		}
		return minColumns;
	}

	/**
	 * \brief Compute cosine of angles between all row vectors in two matrices.
	 */
	static RealMatrix cosAngles(RealMatrix const & m1, RealMatrix const & m2)
	{
		RealMatrix c = prod(m1, trans(m2));
		for(std::size_t i = 0; i < c.size1(); ++i)
		{
			row(c, i) /= norm_2(row(m1, i));
		}
		return c;
	}

	template <typename Archive>
	void serialize(Archive & archive)
	{
		archive & BOOST_SERIALIZATION_NVP(m_alpha);
		archive & BOOST_SERIALIZATION_NVP(m_maxIters);
	}

	double m_alpha;
	std::size_t m_maxIters;
};

} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_REFERENCE_VECTOR_GUIDED_SELECTION_H
