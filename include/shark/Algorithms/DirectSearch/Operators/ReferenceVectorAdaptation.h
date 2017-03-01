#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H

#include <shark/Algorithms/DirectSearch/Operators/Selection/ReferenceVectorGuidedSelection.h>

namespace shark {

void referenceVectorAdaptation(
	double const f_r,
	std::vector<shark::Individual<RealVector, RealVector>> const & population,
	RealMatrix & referenceVectors,
	RealMatrix const & initialReferenceVectors,
	std::size_t const curIteration, std::size_t const maxIteration)
{
	typedef ReferenceVectorGuidedSelection rvgs;
	const std::size_t k = curIteration % static_cast<std::size_t>(std::ceil(f_r * maxIteration));
	if(k == 0)
	{
		RealMatrix f = rvgs::extractPopulationFitness(population);
		RealVector diffMinMaxFitness(f.size2());
		for(std::size_t i = 0; i < f.size2(); ++i)
		{
			diffMinMaxFitness[i] = max(column(f, i)) - min(column(f, i));
		}
		for(std::size_t i = 0; i < referenceVectors.size1(); ++i)
		{
			auto v = row(initialReferenceVectors, i) * diffMinMaxFitness;
			row(referenceVectors, i) = v / norm_2(v);
		}
	}
}

} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H
