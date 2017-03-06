#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H

#include <shark/Algorithms/DirectSearch/Operators/Selection/ReferenceVectorGuidedSelection.h>

namespace shark {

void referenceVectorAdaptation(
	std::vector<shark::Individual<RealVector, RealVector>> const & population,
	RealMatrix & referenceVectors,
	RealMatrix const & initialReferenceVectors)
{
	typedef ReferenceVectorGuidedSelection rvgs;
	RealMatrix f = rvgs::extractPopulationFitness(population);
	RealVector diff(f.size2());
	for(std::size_t i = 0; i < f.size2(); ++i)
	{
		diff[i] = max(column(f, i)) - min(column(f, i));
	}
	referenceVectors = initialReferenceVectors * repeat(diff, initialReferenceVectors.size1());
	for(std::size_t i = 0; i < referenceVectors.size1(); ++i)
	{
		row(referenceVectors, i) /= norm_2(row(referenceVectors, i));
	}
}

} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H
