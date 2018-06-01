//===========================================================================
/*!
 *
 *
 * \brief		Implements the RVEA algorithm.
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

#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_RVEA
#define SHARK_ALGORITHMS_DIRECT_SEARCH_RVEA

#include <shark/Core/DLLSupport.h>
#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/Operators/ReferenceVectorAdaptation.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/ReferenceVectorGuidedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Lattice.h>

namespace shark {

/// \brief Implements the RVEA algorithm.
///
/// Implementation of the RVEA algorithm from the following paper:
/// R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, “A reference vector guided
/// evolutionary algorithm for many-objective optimization,” IEEE Transactions on
/// Evolutionary Computation, Vol 20, Issue 5, October 2016
/// http://dx.doi.org/10.1109/TEVC.2016.2519378
/// \ingroup multidirect
class RVEA : public AbstractMultiObjectiveOptimizer<RealVector>
{
public:
	SHARK_EXPORT_SYMBOL RVEA(random::rng_type & rng = random::globalRng);

	std::string name() const{
		return "RVEA";
	}

	double crossoverProbability() const{
		return m_crossoverProbability;
	}

	double & crossoverProbability(){
		return m_crossoverProbability;
	}

	double nm() const{
		return m_mutation.m_nm;
	}

	double & nm(){
		return m_mutation.m_nm;
	}

	double nc() const{
		return m_crossover.m_nc;
	}

	double & nc(){
		return m_crossover.m_nc;
	}

	double alpha() const{
		return m_selection.m_alpha;
	}

	double & alpha(){
		return m_selection.m_alpha;
	}

	double adaptationFrequency() const{
		return m_adaptParam;
	}

	double & adaptationFrequency(){
		return m_adaptParam;
	}

	/// \brief Size of parent population and number of reference vectors.
	///
	/// In the RVEA algorithm, the exact mu value is determined by the
	/// simplex-lattice design (Lattice.h), so the user cannot set it directly.
	/// Instead, one must set the approxMu() value, which will be used as a
	/// parameter in the lattice.  If one wants to know the exact mu value, set
	/// approxMu() to RVEA::suggestMu(n, m) where n is the objective dimension
	/// and m is the approximate mu.  Then the actual mu value will not be
	/// changed in the initialization.
	std::size_t mu() const{
		return m_mu;
	}
	
	std::size_t numInitPoints() const{
		return m_mu;
	}

	std::size_t approxMu() const{
		return m_approxMu;
	}

	std::size_t & approxMu(){
		return m_approxMu;
	}

	RealMatrix referenceVectors() const{
		return m_referenceVectors;
	}

	RealMatrix & referenceVectors(){
		return m_referenceVectors;
	}

	RealMatrix initialReferenceVectors() const{
		return m_adaptation.m_initVecs;
	}

	std::size_t maxIterations() const{
		return m_selection.m_maxIters;
	}

	std::size_t & maxIterations(){
		return m_selection.m_maxIters;
	}

	/// \brief True if the reference vectors will be adapted.
	///
	/// Returns true if the algorithm will adapt the unit reference vectors in
	/// the current iteration.  This is controlled by the adaptationFreqency()
	/// parameter; a value of, e.g., 0.2 will make the algorithm readapt
	/// reference vectors every 20% of the iteration. Running the algorithm for
	/// 1000 iterations will then make it readapt on iteration 0, 200, 400, etc.
	bool willAdaptReferenceVectors() const{
		return m_curIteration % static_cast<std::size_t>(
			std::ceil(adaptationFrequency() * m_selection.m_maxIters)
			) == 0;
	}

	template <typename Archive>
	void serialize(Archive & archive){
#define S(var) archive & BOOST_SERIALIZATION_NVP(var)
		S(m_crossoverProbability);
		S(m_mu);
		S(m_approxMu);
		S(m_parents);
		S(m_best);
		S(m_crossover);
		S(m_mutation);
		S(m_adaptParam);
		S(m_curIteration);
		S(m_referenceVectors);
		S(m_referenceVectorMinAngles);
		S(m_selection);
		S(m_adaptation);
#undef S
	}

	using AbstractMultiObjectiveOptimizer<RealVector >::init;
	SHARK_EXPORT_SYMBOL void init(
		ObjectiveFunctionType const& function,
		std::vector<SearchPointType> const & initialSearchPoints
	);
	SHARK_EXPORT_SYMBOL void step(
		ObjectiveFunctionType const & function
	);
	SHARK_EXPORT_SYMBOL static std::size_t suggestMu(
		std::size_t n, std::size_t const approx_mu);
protected:
	typedef shark::Individual<RealVector, RealVector> IndividualType;
	SHARK_EXPORT_SYMBOL void doInit(
		std::vector<SearchPointType> const & initialSearchPoints,
		std::vector<ResultType> const & functionValues,
		RealVector const & lowerBounds,
		RealVector const & upperBounds,
		std::size_t const approx_mu,
		double const nm,
		double const nc,
		double const crossover_prob,
		double const alph,
		double const fr,
		std::size_t const max_iterations,
		std::vector<Preference> const & referenceVectorsPreferences = std::vector<Preference>()
	);

	SHARK_EXPORT_SYMBOL std::vector<IndividualType> generateOffspring() const;
	SHARK_EXPORT_SYMBOL void updatePopulation(
		std::vector<IndividualType> const & offspringvec
	);

	std::vector<IndividualType> m_parents;

private:
	random::rng_type * m_rng;
	double m_crossoverProbability; ///< Probability of crossover happening.
	/// \brief Size of parent population
	///
	/// It is also the number of reference vectors.
	std::size_t m_mu;
	/// \brief The "approximate" value of mu that the user asked for.
	///
	/// The actual value of mu is determined via the simplex-lattice design for
	/// the unit reference vectors.  It will always be the same as
	/// RVEA::suggestMu(n, m_approxMu) where n is the number of objectives.
	std::size_t m_approxMu;
	SimulatedBinaryCrossover<SearchPointType> m_crossover;
	PolynomialMutator m_mutation;
	/// \brief Hyperparameter controlling reference vector adaptation rate.
	///
	/// A value of 0.2 makes the algorithm adapt the reference vectors every 20%
	/// of the iterations. If the algorithm runs for a total of 1000 iterations,
	/// they will be readjusted on iteration 0, 200, 400, etc.  Is is called
	/// "f_r" in the paper.
	double m_adaptParam;
	/// \brief Current iteration of the algorithm.
	///
	/// The algorithm maintains knowledge of how long it has been running as
	/// this is required by parts of the RVEA algorithm.
	std::size_t m_curIteration;

	/// \brief The active set of unit reference vectors.
	RealMatrix m_referenceVectors;
	/// \brief Contains the minimum angles between reference vectors.
	///
	/// For each reference vector i, position i in this vector is the smallest
	/// angle between reference vector i and all the other reference vectors.
	RealVector m_referenceVectorMinAngles;
	ReferenceVectorGuidedSelection<IndividualType> m_selection;
	ReferenceVectorAdaptation<IndividualType> m_adaptation;
};

} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_RVEA
