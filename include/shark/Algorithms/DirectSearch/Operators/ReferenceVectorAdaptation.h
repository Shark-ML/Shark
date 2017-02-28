#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H

namespace shark {

struct ReferenceVectorAdaptation
{
	typedef shark::Individual<RealVector, RealVector> IndividualType;
	void operator()(
		double const f_r,
		std::vector<IndividualType> const & population,
		RealMatrix & referenceVectors,
		RealMatrix const & initialReferenceVectors,
		std::size_t const curIteration, std::size_t const maxIteration)
	{
		const double t = static_cast<double>(curIteration);
		const double t_max static_cast<double>(maxIteration);
		if(std::fmod(t / t_max) == 0)
		{
			RealMatrix f = transpose(
				ReferenceVectorGuidedSelection::extractPopulationFitness(
					population));
			RealVector diffMinMaxFitness(f.size1());
			for(std::size_t i = 0; i < f.size1(); ++i)
			{
				diffMinMaxFitness[i] = max(row(f, i)) - min(row(f, i));
			}
			for(std::size_t i = 0; i < referenceVectors.size1(); ++i)
			{
				auto v = row(initialReferenceVectors, i) * diffMinMaxFitness;
				row(referenceVectors, i) = v / norm_2(v);
			}
		}
	}
}

} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_REFERENCE_VECTOR_ADAPTATION_H
