/*!
 *
 *
 * \brief       AGE.h
 *
 *
 *
 * \author      T.Voss
 * \date        2011
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 *
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_AGE_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_AGE_H

#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/Algorithms/DirectSearch/TypedIndividual.h>
// MOO specific stuff
#include <shark/Algorithms/DirectSearch/FastNonDominatedSort.h>
#include <shark/Algorithms/DirectSearch/Indicators/AdditiveEpsilonIndicator.h>
#include <shark/Algorithms/DirectSearch/Indicators/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>

// AGE specific stuff
#include <shark/Algorithms/DirectSearch/Operators/Selection/BinaryTournamentSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>

namespace shark {

namespace detail {

namespace age {
/**
 * \namespace Internal namespace of the AGE.
 */

/**
 * \brief The individual type of the AGE.
 */
typedef TypedIndividual<RealVector> Individual;

/**
 * \brief The population type of the AGE.
 */
typedef std::vector<Individual> Population;
}

/**
 * \brief Implements the AGE.
 *
 * Please see the following papers for further reference:
 *	- Bringmann, Friedrich, Neumann, Wagner. Approximation-Guided Evolutionary Multi-Objective Optimization. IJCAI '11.
 */
class AGE {
protected:
	/** \cond */
	struct AdditiveEpsilonIndicator {
		static double calc(const age::Individual &a, const age::Individual &b)
		{
			return max(a.fitness(tag::PenalizedFitness())-b.fitness(tag::PenalizedFitness()));
		}
	};

	struct CacheElement {
		// age::Population::const_iterator m_p1;
		std::size_t m_p1;
		double m_a1;
		// age::Population::const_iterator m_p2;
		std::size_t m_p2;
		double m_a2;
	};

	std::vector< CacheElement > preProcess(const age::Population &archive, const age::Population &pop) const
	{
		std::vector< CacheElement > result(archive.size());

		for (std::size_t i = 0; i < archive.size(); i++) {
			result[ i ].m_a1 = result[ i ].m_a2 = std::numeric_limits<double>::max();
			for (std::size_t j = 0; j < pop.size(); j++) {

				if (AdditiveEpsilonIndicator::calc(pop[j], archive[i]) < result[ i ].m_a1) {
					result[i].m_p1 = j;
					result[i].m_a1 = AdditiveEpsilonIndicator::calc(pop[j], archive[i]);
				}
			}

			for (std::size_t j = 0; j < pop.size(); j++) {
				if (result[ i ].m_p1 == j)
					continue;
				if (AdditiveEpsilonIndicator::calc(pop[j], archive[i]) < result[ i ].m_a2) {
					result[i].m_p2 = j;
					result[i].m_a2 = AdditiveEpsilonIndicator::calc(pop[j], archive[i]);
				}
			}
		}
		return result;
	}

	double beta(std::size_t p) const
	{
		double result = -std::numeric_limits<double>::max();
		for (std::vector< CacheElement >::const_iterator itt = m_cache.begin(); itt != m_cache.end(); ++itt) {
			if (itt->m_p1 == p)
				result = std::max(result, itt->m_a2);
		}

		return(result);
	}

	/** \endcond */

	std::vector< CacheElement > m_cache;

	age::Population m_archive; ///< Population of size \f$\mu + 1\f$.
	age::Population m_pop; ///< Population of size \f$\mu + 1\f$.
	unsigned int m_mu; ///< Population size \f$\mu\f$.
	unsigned int m_lambda; ///< Offspring population size \f$\lambda\f$.

	ParetoDominanceComparator< shark::tag::PenalizedFitness > m_pdc; /// Pareto dominance comparator.
	shark::moo::PenalizingEvaluator m_evaluator; ///< Evaluation operator.
	FastNonDominatedSort m_fastNonDominatedSort; ///< Operator that provides Deb's fast non-dominated sort.
	IndicatorBasedSelection<HypervolumeIndicator> m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.
	BinaryTournamentSelection< ParetoDominanceComparator<shark::tag::PenalizedFitness> > m_binaryTournamentSelection; ///< Mating selection operator.
	SimulatedBinaryCrossover< RealVector > m_sbx; ///< Crossover operator.
	PolynomialMutator m_mutator; ///< Mutation operator.

	double m_crossoverProbability; ///< Crossover probability.

public:
	/**
	 * \brief The result type of the optimizer, a vector of tuples \f$( \vec x, \vec{f}( \vec{x} )\f$.
	 */
	typedef std::vector< ResultSet< RealVector, RealVector > > SolutionSetType;

	/**
	 * \brief Default c'tor.
	 */
	AGE()
	{
		init();
	}

	/**
	 * \brief Returns the name of the algorithm.
	 */
	std::string name() const
	{
		return("AGE1");
	}

	/**
	 * \brief Stores/loads the algorithm's state.
	 * \tparam Archive The type of the archive.
	 * \param [in,out] archive The archive to use for loading/storing.
	 * \param [in] version Currently unused.
	 */
	template<typename Archive>
	void serialize(Archive &archive, const unsigned int version)
	{
		archive &m_pop;  ///< Population of size \f$\mu + 1\f$.
		archive &m_mu;  /// Population size \f$\mu\f$.

		archive &m_evaluator;  ///< Evaluation operator.
		archive &m_fastNonDominatedSort;  ///< Operator that provides Deb's fast non-dominated sort.
		archive &m_selection;  ///< Selection operator relying on the (contributing) hypervolume indicator.
		archive &m_sbx;  ///< Crossover operator.
		archive &m_mutator;  ///< Mutation operator.

		archive &m_crossoverProbability;  ///< Crossover probability.
	}

	/**
	 * \brief Initializes the algorithm from a configuration-tree node.
	 *
	 * The following sub keys are recognized:
	 *	- Mu, type: unsigned int, default value: 100.
	 *	- Lambda, type: unsigned int, default value: 100.
	 *	- CrossoverProbability, type: double, default value: 0.8.
	 *	- NC, type: double, default value: 10.
	 *	- NM, type: double; default value: 20.
	 *
	 * \param [in] node The configuration tree node.
	 */
	void configure(const PropertyTree &node)
	{
		init(
		    node.get("Mu", 100),
		    node.get("Lambda", 100),
		    node.get("CrossoverProbability", 0.9),
		    node.get("NC", 20.0),
		    node.get("NM", 20.0)
		);
	}

	/**
	 * \brief Initializes the algorithm.
	 * \param [in] mu The population size.
	 * \param [in] lambda The offspring population size.
	 * \param [in] pc Crossover probability, default value: 0.8.
	 * \param [in] nc Parameter of the simulated binary crossover operator, default value: 10.0.
	 * \param [in] nm Parameter of the mutation operator, default value: 20.0.
	 */
	void init(unsigned int mu = 100,
	          unsigned int lambda = 1,
	          double pc = 0.9,
	          double nc = 20.0,
	          double nm = 20.0
	         )
	{
		m_mu = mu;
		m_lambda = lambda;
		m_crossoverProbability = pc;
		m_sbx.m_nc = nc;
		m_mutator.m_nm = nm;

		m_selection.setMu(m_mu);
	}

	/**
	 * \brief Initializes the algorithm for the supplied objective function.
	 * \tparam ObjectiveFunction The type of the objective function,
	 * needs to adhere to the concept of an AbstractObjectiveFunction.
	 * \param [in] f The objective function.
	 * \param [in] sp An initial search point.
	 */
	template<typename Function>
	void init(const Function &f, const RealVector &sp)
	{

		(void) sp;

		m_pop.resize(m_mu + 1);

		std::size_t noObjectives = 0;

		for (age::Population::iterator it = m_pop.begin(); it != m_pop.end(); ++it) {
			it->age()=0;

			f.proposeStartingPoint(**it);
			boost::tuple< typename Function::ResultType, typename Function::ResultType > result = m_evaluator(f, **it);
			it->fitness(shark::tag::PenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >(result);
			it->fitness(shark::tag::UnpenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >(result);

			noObjectives = std::max(noObjectives, it->fitness(shark::tag::PenalizedFitness()).size());
			it->setNoObjectives(noObjectives);
		}

		m_sbx.init(f);
		m_mutator.init(f);

		m_fastNonDominatedSort(m_pop);
		m_archive.clear();
		for (age::Population::iterator it = m_pop.begin(); it != m_pop.end(); ++it) {
			if (it->rank() > 1)
				continue;
			m_archive.push_back(*it);
		}

		m_cache = preProcess(m_archive, m_pop);
	}

	/**
	 * \brief Executes one iteration of the algorithm.
	 * \tparam The type of the objective to iterate upon.
	 * \param [in] f The function to iterate upon.
	 * \returns The Pareto-set/-front approximation after the iteration.
	 */
	template<typename Function>
	SolutionSetType step(const Function &f)
	{

		age::Individual mate1(*m_binaryTournamentSelection(m_pop.begin(), m_pop.begin() + m_mu));
		age::Individual mate2(*m_binaryTournamentSelection(m_pop.begin(), m_pop.begin() + m_mu));

		if (Rng::coinToss(m_crossoverProbability)) {
			m_sbx(mate1, mate2);
		}

		if (Rng::coinToss()) {
			m_mutator(mate2);
			m_pop.back() = mate2;
		} else  {
			m_mutator(mate1);
			m_pop.back() = mate1;
		}

		boost::tuple< typename Function::ResultType, typename Function::ResultType > result = m_evaluator(f, *m_pop.back());
		m_pop.back().fitness(shark::tag::PenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >(result);
		m_pop.back().fitness(shark::tag::UnpenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >(result);


		// Iterate the archive.
		ParetoDominanceComparator< shark::tag::PenalizedFitness > pdc;
		bool dominated = false;
		age::Population::iterator it = m_archive.begin();
		while (it != m_archive.end()) {
			switch (pdc(m_pop.back(), *it)) {
			case ParetoDominanceComparator< shark::tag::PenalizedFitness >::A_STRICTLY_DOMINATES_B:
			case ParetoDominanceComparator< shark::tag::PenalizedFitness >::A_WEAKLY_DOMINATES_B:
				it = m_archive.erase(it);
				break;
			case ParetoDominanceComparator< shark::tag::PenalizedFitness >::B_STRICTLY_DOMINATES_A:
			case ParetoDominanceComparator< shark::tag::PenalizedFitness >::B_WEAKLY_DOMINATES_A:
				dominated = true;
				break;
			case ParetoDominanceComparator< shark::tag::PenalizedFitness >::A_EQUALS_B:
			case ParetoDominanceComparator< shark::tag::PenalizedFitness >::TRADE_OFF:
				++it;
				break;
			}

			if (dominated)
				break;
		}

		if (!dominated)
			m_archive.push_back(m_pop.back());

		m_cache = preProcess(m_archive, m_pop);
		std::vector<double> b(m_pop.size(), -std::numeric_limits< double >::max());

		for (std::size_t i = 0; i < m_archive.size(); i++) {
			b[ m_cache[ i ].m_p1 ] = std::max(m_cache[ i ].m_a2, b[ m_cache[ i ].m_p1 ]);
		}

		std::vector<double>::iterator maxBeta = std::min_element(b.begin(), b.end());
		m_pop[ std::distance(b.begin(), maxBeta) ] = m_pop.back();


		m_fastNonDominatedSort(m_pop);

		SolutionSetType solutionSet;
		for (age::Population::iterator it = m_pop.begin(); it != m_pop.begin() + m_mu; ++it) {
			solutionSet.push_back(shark::makeResultSet(*(*it), it->fitness(shark::tag::UnpenalizedFitness()))) ;
		}

		return(solutionSet);
	}
};
}
}
#endif