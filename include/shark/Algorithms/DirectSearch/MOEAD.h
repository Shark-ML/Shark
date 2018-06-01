//===========================================================================
/*!
 *
 *
 * \brief		Implements the MOEA/D algorithm.
 *
 * \author		Bjoern Bugge Grathwohl
 * \date		February 2017
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

#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_MOEAD
#define SHARK_ALGORITHMS_DIRECT_SEARCH_MOEAD

#include <shark/Core/DLLSupport.h>
#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Lattice.h>

namespace shark {

/// \brief Implements the MOEA/D algorithm.
///
/// Implementation of the MOEA/D algorithm from the following paper:
/// Q. Zhang and H. Li, MOEA/D: a multi-objective evolutionary algorithm based
/// on decomposition, IEEE Transactions on Evolutionary Computation, vol. 11,
/// no. 6, pp. 712 - 731, 2007
/// DOI: 10.1109/TEVC.2007.892759
/// \ingroup multidirect
class MOEAD : public AbstractMultiObjectiveOptimizer<RealVector>
{
public:
	SHARK_EXPORT_SYMBOL MOEAD(random::rng_type & rng = random::globalRng);

	std::string name() const{
		return "MOEA/D";
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

	std::size_t mu() const{
		return m_mu;
	}

	std::size_t & mu(){
		return m_mu;
	}
	
	/// \brief The number of points used for initialization of the algorithm.
	std::size_t numInitPoints() const{
		return m_mu;
	}

	std::size_t neighbourhoodSize() const{
		return m_neighbourhoodSize;
	}

	std::size_t & neighbourhoodSize(){
		return m_neighbourhoodSize;
	}

	template <typename Archive>
	void serialize(Archive & archive)
	{
		archive & BOOST_SERIALIZATION_NVP(m_crossoverProbability);
		archive & BOOST_SERIALIZATION_NVP(m_mu);
		archive & BOOST_SERIALIZATION_NVP(m_parents);
		archive & BOOST_SERIALIZATION_NVP(m_best);
		archive & BOOST_SERIALIZATION_NVP(m_weights);
		archive & BOOST_SERIALIZATION_NVP(m_neighbourhoods);
		archive & BOOST_SERIALIZATION_NVP(m_neighbourhoodSize);
		archive & BOOST_SERIALIZATION_NVP(m_bestDecomposedValues);
		archive & BOOST_SERIALIZATION_NVP(m_crossover);
		archive & BOOST_SERIALIZATION_NVP(m_mutation);
		archive & BOOST_SERIALIZATION_NVP(m_curParentIndex);
	}

	using AbstractMultiObjectiveOptimizer<RealVector >::init;
	SHARK_EXPORT_SYMBOL void init(
		ObjectiveFunctionType const& function,
		std::vector<SearchPointType> const & initialSearchPoints
	);
	SHARK_EXPORT_SYMBOL void step(ObjectiveFunctionType const & function);
protected:
	typedef shark::Individual<RealVector, RealVector> IndividualType;
	SHARK_EXPORT_SYMBOL void doInit(
		std::vector<SearchPointType> const & initialSearchPoints,
		std::vector<ResultType> const & functionValues,
		RealVector const & lowerBounds,
		RealVector const & upperBounds,
		std::size_t const mu,
		double const nm,
		double const nc,
		double const crossover_prob,
		std::size_t const neighbourhoodSize,
		std::vector<Preference> const & weightVectorPreferences = std::vector<Preference>()
	);
	// Make me an offspring...
	SHARK_EXPORT_SYMBOL std::vector<IndividualType> generateOffspring() const;
	SHARK_EXPORT_SYMBOL void updatePopulation(std::vector<IndividualType> const & offspringvec);

	std::vector<IndividualType> m_parents;

private:
	random::rng_type * mpe_rng;
	double m_crossoverProbability; ///< Probability of crossover happening.
	std::size_t m_mu; ///< Size of parent population and the "N" from the paper

	std::size_t m_curParentIndex;

	/// \brief Number of neighbours for each candidate to consider.
	///
	/// This is the "T" from the paper.
	std::size_t m_neighbourhoodSize;
	RealMatrix m_weights; ///< The weight vectors. These are all the lambdas from the paper
	
	/// \brief Row n stores the indices of the T closest weight vectors. 
	///
	/// This is the "B" function from the paper.
	UIntMatrix m_neighbourhoods; 
	RealVector m_bestDecomposedValues; ///< The "z" from the paper.

	SimulatedBinaryCrossover<SearchPointType> m_crossover;
	PolynomialMutator m_mutation;
};


} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_MOEAD
