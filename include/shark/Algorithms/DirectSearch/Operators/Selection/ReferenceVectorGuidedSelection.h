#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_REFERENCE_VECTOR_GUIDED_SELECTION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_REFERENCE_VECTOR_GUIDED_SELECTION_H

namespace shark {

struct ReferenceVectorGuidedSelection
{
	typedef shark::Individual<RealVector, RealVector> IndividualType;

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
		const std::size_t populationSize = fitness.size1();
		const std::size_t objectiveCount = fitness.size2();

		// Objective value translation
		// line 4
		const RealVector minFitness = minFitnessValues(fitness);
		// line 5-7
		fitness -= repeat(minFitness, fitness.size1());

		// Population partition
		// line 9-13
		RealMatrix cosAngles = prod(fitness, trans(referenceVectors));
		SIZE_CHECK(cosAngles.size1() == populationSize);
		SIZE_CHECK(cosAngles.size2() == groupCount);
		for(std::size_t i = 0; i < cosAngles.size1(); ++i)
		{
			row(cosAngles, i) /= norm_2(row(fitness, i));
		}
		// line 14-17
		// Contains indices
		std::vector<std::set<std::size_t>> subGroups(groupCount);
		for(std::size_t i = 0; i < cosAngles.size1(); ++i)
		{
			const std::size_t k = std::distance(
				row(cosAngles, i).begin(),
				std::max_element(row(cosAngles, i).begin(),
				                 row(cosAngles, i).end()));
			subGroups[k].insert(i);
		}

		// Angle-penalized distance (APD) calculation
		// line 19-23
		const RealMatrix angles = acos(cosAngles);
		const RealVector gammas = leastAngles(referenceVectors);
		const double theta = objectiveCount *
			std::pow(static_cast<double>(curIteration) /
					 static_cast<double>(maxIteration), alpha);
		RealMatrix apDists(populationSize, groupCount);
		for(std::size_t j = 0; j < groupCount; ++j)
		{
			for(std::size_t i : subGroups[j])
			{
				apDists(i, j) = 1 + theta * angles(i, j) / gammas[j];
				apDists(i, j) *= norm_2(row(fitness, i));
			}
		}

		// Elitism selection
		for(auto & p : population)
		{
			p.selected() = false;
		}
		std::set<std::size_t> was_selected;
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
//			std::cout << "selected idx " << min_idx << " for group " << j << "\n";
			population[min_idx].selected() = true;
//			was_selected.insert(min_idx);
		}
//		SIZE_CHECK(was_selected.size() == groupCount);
	}

	RealVector leastAngles(RealMatrix const & referenceVectors)
	{
		RealVector la(referenceVectors.size1());
		for(std::size_t i = 0; i < referenceVectors.size1(); ++i)
		{
			la[i] = leastAngleFrom(i, referenceVectors);
		}
		return la;
	}

	static double leastAngleFrom(
		std::size_t const idx, RealMatrix const & referenceVectors)
	{
		double leastAngle = 1e10;
		for(std::size_t i = 0; i < referenceVectors.size1(); ++i)
		{
			if(i == idx)
			{
				continue;
			}
			// The reference vectors all have length 1.
			double thisAngle =  std::acos(
				inner_prod(row(referenceVectors, i),
						   row(referenceVectors, idx)));
			leastAngle = std::min(leastAngle, thisAngle);
		}
		return leastAngle;
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
