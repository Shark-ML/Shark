#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H

#include <shark/Algorithms/DirectSearch/Operators/Selection/ReferenceVectorGuidedSelection.h>

namespace shark {

struct ReferenceVectorAdaptation
{
	typedef shark::Individual<RealVector, RealVector> IndividualType;

	void operator()(
		std::vector<IndividualType> const & population,
		RealMatrix & referenceVectors,
		RealVector & minAngles)
	{
		typedef ReferenceVectorGuidedSelection rv;
		const RealMatrix f = rv::extractPopulationFitness(population);
		RealVector diff(f.size2());
		for(std::size_t i = 0; i < f.size2(); ++i)
		{
			diff[i] = max(column(f, i)) - min(column(f, i));
		}
		referenceVectors = m_initVecs * repeat(diff, m_initVecs.size1());
		for(std::size_t i = 0; i < referenceVectors.size1(); ++i)
		{
			row(referenceVectors, i) /= norm_2(row(referenceVectors, i));
		}
		updateAngles(referenceVectors, minAngles);
	}

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

    RealMatrix m_initVecs;
};

} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H
