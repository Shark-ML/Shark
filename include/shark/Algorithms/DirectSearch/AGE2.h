//===========================================================================
/*!
 * 
 *
 * \brief       AGE2.h
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
//===========================================================================
#pragma once

#include <shark/Algorithms/AbstractOptimizer.h>

#include <shark/Algorithms/DirectSearch/TypedIndividual.h>
#include <shark/Core/Traits/OptimizerTraits.h>
// MOO specific stuff
#include <shark/Algorithms/DirectSearch/ParetoDominanceComparator.h>
#include <shark/Algorithms/DirectSearch/FastNonDominatedSort.h>
#include <shark/Algorithms/DirectSearch/Indicators/AdditiveEpsilonIndicator.h>
#include <shark/Algorithms/DirectSearch/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/RankShareComparator.h>

// AGE specific stuff
#include <shark/Algorithms/DirectSearch/Operators/Selection/BinaryTournamentSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

#include <shark/LinAlg/LinAlg.h>

namespace shark {

namespace detail {

namespace age2 {
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
 * \brief Implements the AGE2.
 *
 * Please see the following papers for further reference:
 *	- Bringmann, Friedrich, Neumann, Wagner. Approximation-Guided Evolutionary Multi-Objective Optimization. IJCAI '11.
 */
class AGE2 {
protected:

	/** \cond */
	struct AdditiveEpsilonIndicator {
		static double calc(const age2::Individual &a, const age2::Individual &b) {
			double result = -std::numeric_limits<double>::max();
			for (unsigned int i = 0; i < a.fitness(tag::PenalizedFitness()).size(); i++) {
				result = std::max(result, b.fitness(tag::PenalizedFitness())[ i ] - a.fitness(tag::PenalizedFitness())[i]);
			}
			return(result);
		}
	};

	struct CacheElement {
		age2::Population::const_iterator m_p1;
		double m_a1;
		age2::Population::const_iterator m_p2;
		double m_a2;
	};

	struct MinElement {

		MinElement(const age2::Individual &a) : m_a(a) {
		}

		bool operator()(const age2::Individual &x, const age2::Individual &y) const {
			return(AdditiveEpsilonIndicator::calc(x, m_a) < AdditiveEpsilonIndicator::calc(y, m_a));
		}

		const age2::Individual &m_a;

	};

	std::vector< CacheElement > preProcess(const age2::Population &archive, const age2::Population &pop) const {
		std::vector< CacheElement > result(archive.size());

		for (unsigned int i = 0; i < archive.size(); i++) {
			MinElement me(archive[i]);
			result[i].m_p1 = result[i].m_p2 = pop.end();
			for (age2::Population::const_iterator it = pop.begin(); it != pop.end(); ++it) {
				if (result[i].m_p1 == pop.end()) {
					result[i].m_p1 = it;
					continue;
				}

				if (me(*result[i].m_p1, *it)) {
					result[i].m_p1 = it;
					result[i].m_a1 = AdditiveEpsilonIndicator::calc(*it, archive[i]);
				}
			}

			for (age2::Population::const_iterator it = pop.begin(); it != pop.end(); ++it) {
				if (it == result[i].m_p1)
					continue;

				if (result[i].m_p2 == pop.end()) {
					result[i].m_p2 = it;
					continue;
				}

				if (me(*result[i].m_p2, *it)) {
					result[i].m_p2 = it;
					result[i].m_a2 = AdditiveEpsilonIndicator::calc(*it, archive[i]);
				}
			}


		}
		return(result);
	}
	/** \endcond */
	std::vector< CacheElement > m_cache;

	age2::Population m_archive; ///< Population of size \f$\mu + 1\f$.
	age2::Population m_pop; ///< Population of size \f$\mu + 1\f$.
	unsigned int m_mu; ///< Population size \f$\mu\f$.
	unsigned int m_lambda; ///< Offspring population size \f$\lambda\f$.

	shark::moo::PenalizingEvaluator m_evaluator; ///< Evaluation operator.
	RankShareComparator rsc; ///< Comparator for individuals based on their multi-objective rank and share.
	FastNonDominatedSort m_fastNonDominatedSort; ///< Operator that provides Deb's fast non-dominated sort.
	IndicatorBasedSelection<HypervolumeIndicator> m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.
	BinaryTournamentSelection< RankShareComparator > m_binaryTournamentSelection; ///< Mating selection operator.
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
	AGE2() : m_binaryTournamentSelection(rsc) {
		init();
	}

	/**
	 * \brief Returns the name of the algorithm.
	 */
	std::string name() const {
		return("AGE2");
	}

	/**
	 * \brief Stores/loads the algorithm's state.
	 * \tparam Archive The type of the archive.
	 * \param [in,out] archive The archive to use for loading/storing.
	 * \param [in] version Currently unused.
	 */
	template<typename Archive>
	void serialize(Archive &archive, const unsigned int version) {
		archive &m_pop;  ///< Population of size \f$\mu + 1\f$.
		archive &m_mu;  /// Population size \f$\mu\f$.

		archive &m_evaluator;  ///< Evaluation operator.
		archive &rsc;  ///< Comparator for individuals based on their multi-objective rank and share.
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
	void configure(const PropertyTree &node) {
		init(
		    node.get("Mu", 100),
		    node.get("Lambda", 100),
		    node.get("CrossoverProbability", 0.8),
		    node.get("NC", 10.0),
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
	void init(unsigned int mu = 300,
	        unsigned int lambda = 100,
	        double pc = 0.9,
	        double nc = 20.0,
	        double nm = 20.0) {
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
	 * \param [in] f The objective function
	 */
	template<typename Function>
	void init(const Function &f) {
		m_pop.resize(m_mu + 1);
		m_selection.setNoObjectives(f.noObjectives());
		
		for (age2::Population::iterator it = m_pop.begin(); it != m_pop.end(); ++it) {
			it->age()=0;
			it->setNoObjectives(f.noObjectives());
			f.proposeStartingPoint(**it);
			boost::tuple< typename Function::ResultType, typename Function::ResultType > result = m_evaluator(f, **it);
			it->fitness(shark::tag::PenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >(result);
			it->fitness(shark::tag::UnpenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >(result);
		}
		
		m_sbx.init(f);
		m_mutator.init(f);
		
		m_fastNonDominatedSort(m_pop);
		m_archive.clear();
		for (age2::Population::iterator it = m_pop.begin(); it != m_pop.end(); ++it) {
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
	SolutionSetType step(const Function &f) {
		int maxIdx = 0;
		for (unsigned int i = 0; i < m_pop.size(); i++) {
			if (m_pop[i].rank() != 1)
				break;
			maxIdx = i;
		}

		age2::Individual mate1(m_pop[Rng::discrete(0, maxIdx)]);
		age2::Individual mate2(m_pop[Rng::discrete(0, maxIdx)]);

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
		age2::Population::iterator it = m_archive.begin();
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
				continue;
				break;
			}

			if (dominated)
				break;
		}

		if (!dominated)
			m_archive.push_back(m_pop.back());

		//std::cout << "(II) Evaluation counter: " << f.evaluationCounter() << std::endl;
		m_cache = preProcess(m_archive, m_pop);

		RealVector beta(m_pop.size(), -std::numeric_limits< double >::max());
		RealVector::iterator itb = beta.begin();
		for (it = m_pop.begin(); it != m_pop.end(); ++it, ++itb) {
			it->share() = 0;
			for (std::vector< CacheElement >::iterator itt = m_cache.begin(); itt != m_cache.end(); ++itt) {
				if (itt->m_p1 == it)
					*itb = std::max(*itb, itt->m_a2);
			}
		}

		m_fastNonDominatedSort(m_pop);
		RealVector::iterator maxBeta = std::max_element(beta.begin(), beta.end());
		m_pop[ std::distance(beta.begin(), maxBeta) ].share() = 1;
		std::sort(m_pop.begin(), m_pop.end(), RankShareComparator());
		//std::swap( m_pop[ std::distance( beta.begin(), maxBeta ) ], m_pop.back() );

		/*typedef LocalitySensitiveAdditiveEpsilonIndicator< shark::tag::PenalizedFitness > LSAE;
		  LSAE lsae;
		  std::vector< LSAE::ResultType > indicatorValues( m_pop.size() );
		  for( unsigned int i = 0; i < m_pop.size(); i++ ) {
		  age2::Population p( m_pop );
		  p.erase( p.begin() + i );

		  indicatorValues[ i ] = lsae( p.begin(), p.end(), m_archive.begin(), m_archive.end() );
		  }

		  m_fastNonDominatedSort( m_pop );
		  for( age2::Population::iterator it = m_pop.begin(); it != m_pop.end(); ++it )
		  it->share() = 0;

		  std::vector<
		  LSAE::ResultType
		  >::iterator itr = std::min_element( indicatorValues.begin(), indicatorValues.end() );

		  m_pop[ std::distance( indicatorValues.begin(), itr ) ].share() = 1;

		  std::sort( m_pop.begin(), m_pop.end(), RankShareComparator() );*/


		SolutionSetType solutionSet;
		for (age2::Population::iterator it = m_archive.begin(); it != m_archive.end(); ++it) {
			solutionSet.push_back(shark::makeResultSet(*(*it), it->fitness(shark::tag::UnpenalizedFitness()))) ;
		}

		return(solutionSet);
	}
};
}

/**
 * \brief AGE specialization of optimizer traits.
 */
template<>
struct OptimizerTraits<detail::AGE2> {
	/**
	 * \brief Prints out the configuration options and usage remarks of the algorithm to the supplied stream.
	 * \tparam Stream The type of the stream to output to.
	 * \param [in,out] s The stream to print usage information to.
	 */
	template<typename Stream>
	static void usage(Stream &s) {
		s << "AGE2 usage information:" << std::endl;
		s << "\t Mu, size of the population, default value: \t\t 100" << std::endl;
		s << "\t Lambda, size of the offspring population, default value: \t\t 100" << std::endl;
		s << "\t CrossoverProbability, type: double, default value: \t\t 0.8." << std::endl;
		s << "\t NC, type: double, default value: \t\t 10." << std::endl;
		s << "\t NM, type: double; default value: \t\t 20." << std::endl;
	}
};
}

