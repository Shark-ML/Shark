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
		RealVector const & gammas,
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
		SIZE_CHECK(subGroups.size() == groupCount);
		// Elitism selection
		for(auto & p : population)
		{
			p.selected() = false;
		}
		const double theta = fitness.size2() 
			* std::pow(static_cast<double>(curIteration) / 
			           static_cast<double>(maxIteration), alpha);
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
