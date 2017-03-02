#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_REFERENCE_VECTOR_GUIDED_SELECTION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_REFERENCE_VECTOR_GUIDED_SELECTION_H

#include <shark/LinAlg/Base.h>
#include <shark/Algorithms/DirectSearch/Individual.h>

namespace shark {

struct ReferenceVectorGuidedSelection
{
	typedef shark::Individual<RealVector, RealVector> IndividualType;
	typedef std::set<std::size_t> bag_t;
	

	void operator()(
		double const alpha,
		std::vector<IndividualType> & population,
		RealMatrix const & referenceVectors,
		std::size_t const curIteration, std::size_t const maxIteration)
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
		const RealVector minFitness = minFitnessValues(fitness);
		// line 5-7
		fitness -= repeat(minFitness, fitness.size1());

		// Population partition
		// line 9-13
		const RealMatrix cosA = cosAngles(fitness, referenceVectors);
		const RealMatrix angles = acos(cosA);
		// line 14-17
		const std::vector<bag_t> subGroups = populationPartition(cosA);

		// Angle-penalized distance (APD) calculation
		// line 19-23
		const RealMatrix apDists = anglePenalizedDistance(
			angles, referenceVectors, fitness, subGroups,
			curIteration, maxIteration, alpha);

		// Elitism selection
		for(auto & p : population)
		{
			p.selected() = false;
		}
		// line 25-28
		for(std::size_t j = 0; j < groupCount; ++j)
		{
			std::size_t min_idx = 0;
			double min = 1e10;
			for(std::size_t i : subGroups[j])
			{
				if(apDists(i, j) < min)
				{
					min_idx = i;
					min = apDists(i, j);
				}
			}
			population[min_idx].selected() = true;
		}
	}

	static RealMatrix anglePenalizedDistance(
		RealMatrix const & angles, RealMatrix const & referenceVectors,
		RealMatrix const & fitness, std::vector<bag_t> const & subGroups,
		double const t, double const t_max, 
		double const alpha)
	{
		const RealVector gammas = leastAngles(referenceVectors);
		const std::size_t objCount = fitness.size2();
		const double theta = objCount * std::pow(t / t_max, alpha);
		RealMatrix apDists(angles.size1(), referenceVectors.size1());
		for(std::size_t j = 0; j < referenceVectors.size1(); ++j)
		{
			for(std::size_t i : subGroups[j])
			{
				apDists(i, j) = 1 + theta * angles(i, j) / gammas[j];
				apDists(i, j) *= norm_2(row(fitness, i));
			}
		}
		return apDists;
	}

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

	static RealMatrix cosAngles(
		RealMatrix const & fitness, 
		RealMatrix const & referenceVectors)
	{
		RealMatrix c = prod(fitness, trans(referenceVectors));
		for(std::size_t i = 0; i < c.size1(); ++i)
		{
			row(c, i) /= norm_2(row(fitness, i));
		}
		return c;
	}

	static RealVector leastAngles(RealMatrix const & referenceVectors)
	{
		RealVector la(referenceVectors.size1(), 1e10);
		for(std::size_t i = 0; i < referenceVectors.size1(); ++i)
		{
			for(std::size_t j = 0; j < referenceVectors.size1(); ++j)
			{
				if(j == i) 
				{
					continue;
				}
				double thisAngle =  std::acos(
					inner_prod(row(referenceVectors, j),
					           row(referenceVectors, i)));
				la[i] = std::min(la[i], thisAngle);
			}
		}
		return la;
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

	static RealVector minFitnessValues(RealMatrix const & fitness)
	{
		RealVector minFitness(fitness.size2());
		for(std::size_t i = 0; i < fitness.size2(); ++i)
		{
			minFitness[i] = min(column(fitness, i));
		}
		return minFitness;
	}
};

} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_REFERENCE_VECTOR_GUIDED_SELECTION_H
