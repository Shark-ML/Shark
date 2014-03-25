/*!
 * 
 *
 * \brief       SteadyStateMOCMA.h
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_STEADYSTATEMOCMA
#define SHARK_ALGORITHMS_DIRECT_SEARCH_STEADYSTATEMOCMA

#include <shark/Algorithms/DirectSearch/TypedIndividual.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <shark/Algorithms/DirectSearch/CMA/Chromosome.h>

#include <shark/Algorithms/DirectSearch/Operators/Initializers/CMA/Initializer.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/CMA/Mutator.h>
// MOO specific stuff
#include <shark/Algorithms/DirectSearch/Indicators/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Indicators/AdditiveEpsilonIndicator.h>
#include <shark/Algorithms/DirectSearch/Indicators/LeastContributorApproximator.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Core/ResultSets.h>
#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>
#include <boost/foreach.hpp>

namespace shark {
namespace detail {

/**
 * \brief Implements the \f$(\mu+1)\f$-MO-CMA-ES.
 *
 * Please see the following papers for further reference:
 *	- Igel, Suttorp and Hansen. Steady-state Selection and Efficient Covariance Matrix Update in the Multi-Objective CMA-ES.
 *	- Voﬂ, Hansen and Igel. Improved Step Size Adaptation for the MO-CMA-ES.
 */
template<typename Indicator=HypervolumeIndicator>
struct SteadyStateMOCMA {
	/**
	 * \namespace Internal namespace of the \f$(\mu+1)\f$-MO-CMA-ES.
	 */

	typedef shark::elitist_cma::Chromosome Chromosome;

	/**
	 * \brief The individual type of the \f$(\mu+1)\f$-MO-CMA-ES.
	 */
	typedef TypedIndividual<RealVector,Chromosome> Individual;

	/**
	 * \brief The population type of the \f$(\mu+1)\f$-MO-CMA-ES.
	 */
	typedef std::vector< Individual > Population;

	/**
	 * \brief Individual and chromosome initializer type.
	 */
	typedef shark::elitist_cma::Initializer<Individual,Chromosome,0> Initializer;

	/**
	 * \brief Mutation operator.
	 */
	typedef shark::cma::Variator<Individual,Chromosome,0> Variator;
	/**
	 * \brief Typedef of the algorithm's own type.
	 */
	typedef SteadyStateMOCMA<Indicator> this_type;

	/**
	 * \brief Default parent population size.
	 */
	static std::size_t DEFAULT_MU(){
		return 100;
	}

	/**
	 * \brief Default penalty factor for penalizing infeasible solutions.
	 */
	static double DEFAULT_PENALTY_FACTOR() {
		return 1E-6;
	}

	/**
	 * \brief Default success threshold for step-size adaptation.
	 */
	static double DEFAULT_SUCCESS_THRESHOLD() {
		return 0.44;
	}

	/**
	 * \brief Default notion of success.
	 */
	static const char *DEFAULT_NOTION_OF_SUCCESS(){
		return "IndividualBased";
	}
	
	/**
	 * \brief Default choice for the initial sigma
	 */
	static double DEFAULT_INITIAL_SIGMA() {
		return 1.0;
	}

	/**
	 * \brief The result type of the optimizer, a vector of tuples \f$( \vec x, \vec{f}( \vec{x} )\f$.
	 */
	typedef std::vector< ResultSet< RealVector, RealVector > > SolutionSetType;

	Population m_pop; ///< Population of size \f$\mu+1\f$.
	Initializer m_initializer; ///< Initialization operator.
	Variator m_variator; ///< Mutation operator.
	shark::moo::PenalizingEvaluator m_evaluator; ///< Evaluation operator.

	IndicatorBasedSelection<Indicator> m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.
	
	unsigned int m_mu; ///< Population size \f$\mu\f$.
	bool m_useNewUpdate; ///< Flag for deciding whether the improved step-size adaptation shall be used.
	
	/**
	 * \brief Default c'tor.
	 */
	SteadyStateMOCMA() {
		init();
	}

	/**
	 * \brief Returns the name of the algorithm.
	 */
	std::string name() const {
		return("SteadyStateMOCMA");
	}

	/**
	 * \brief Stores/loads the algorithm's state.
	 * \tparam Archive The type of the archive.
	 * \param [in,out] archive The archive to use for loading/storing.
	 * \param [in] version Currently unused.
	 */
	template<typename Archive>
	void serialize(Archive &archive, const unsigned int version) {
		archive &BOOST_SERIALIZATION_NVP(m_pop);
		archive &BOOST_SERIALIZATION_NVP(m_initializer);
		archive &BOOST_SERIALIZATION_NVP(m_variator);

		archive &BOOST_SERIALIZATION_NVP(m_evaluator);
		archive &BOOST_SERIALIZATION_NVP(m_selection);

		archive &BOOST_SERIALIZATION_NVP(m_mu);
		archive &BOOST_SERIALIZATION_NVP(m_useNewUpdate);
	}

	/**
	 * \brief Initializes the algorithm.
	 * \param [in] mu The population size.
	 * \param [in] penaltyFactor The penalty factor for penalizing infeasible solutions.
	 * \param [in] successThreshold The success threshold \f$p_{\text{thresh}}\f$ for cutting off evolution path updates.
	 * \param [in] notionOfSuccess The notion of success.
	 * \param [in] initialSigma value for the initial choice of sigma.
	 */
	void init(unsigned int mu = this_type::DEFAULT_MU(),
	        double penaltyFactor = this_type::DEFAULT_PENALTY_FACTOR(),
	        double successThreshold = this_type::DEFAULT_SUCCESS_THRESHOLD(),
	        const std::string &notionOfSuccess = this_type::DEFAULT_NOTION_OF_SUCCESS(),
	        double initialSigma = this_type::DEFAULT_INITIAL_SIGMA()
	) {
		m_mu = mu;
		m_selection.setMu(mu);
		m_evaluator.m_penaltyFactor = penaltyFactor;
		m_initializer.m_successThreshold = successThreshold;

		if (notionOfSuccess == "IndividualBased") {
			m_useNewUpdate = false;
		} else if (notionOfSuccess == "PopulationBased") {
			m_useNewUpdate = true;
		}
	}

	/**
	 * \brief Initializes the algorithm from a configuration-tree node.
	 *
	 * The following sub keys are recognized:
	 *	- Mu, type: unsigned int, default value: 100.
	 *	- PenaltyFactor, type: double, default value: \f$10^{-6}\f$.
	 *	- SuccessThreshold, type: double, default value: 0.44.
	 *	- NotionOfSuccess, type: string, default value: IndividualBased.
	 *	- InitialSigma: the initial value of standard deviation of the distribution. default is 1.0.
	 *
	 * \param [in] node The configuration tree node.
	 */
	template<typename PropTree>
	void configure(const PropTree &node) {
		init(
			node.template get<unsigned int>("Mu", this_type::DEFAULT_MU()),
			node.template get<double>("PenaltyFactor", this_type::DEFAULT_PENALTY_FACTOR()),
			node.template get<double>("SuccessThreshold", this_type::DEFAULT_SUCCESS_THRESHOLD()),
			node.template get<std::string>("NotionOfSuccess", this_type::DEFAULT_NOTION_OF_SUCCESS()),
			node.template get<double>("InitialSigma",this_type::DEFAULT_INITIAL_SIGMA())
		);
	}

	/**
	 * \brief Initializes the algorithm for the supplied objective function.
	 * \tparam ObjectiveFunction The type of the objective function,
	 * needs to adhere to the concept of an AbstractObjectiveFunction.
	 * \param [in] f The objective function.
	 * \param [in] sp An initial search point.
	 */
	template<typename ObjectiveFunction>
	void init(const ObjectiveFunction &f, const RealVector &sp = RealVector()) {

		m_pop.resize(m_mu + 1);
		m_initializer.m_searchSpaceDimension = f.numberOfVariables();
		m_initializer.m_noObjectives = f.numberOfObjectives();

		shark::moo::PenalizingEvaluator evaluator;
		BOOST_FOREACH(Individual & ind, m_pop) {
			f.proposeStartingPoint(*ind);
			boost::tuple< typename ObjectiveFunction::ResultType, typename ObjectiveFunction::ResultType > 	result = evaluator(f, *ind);
			ind.fitness(shark::tag::PenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >(result);
			ind.fitness(shark::tag::UnpenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >(result);
			m_initializer(ind);
		}
		m_selection(m_pop);
		std::sort(m_pop.begin(), m_pop.end(), Individual::RankOrdering);
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

		Individual& parent = m_pop[Rng::discrete(0, std::max(0, maxIdx-1))];
		Individual& offspring = m_pop[m_mu];
		offspring.searchPoint() = parent.searchPoint();
		offspring.age() = 0;
		m_variator(offspring);

		shark::moo::PenalizingEvaluator evaluator;
		boost::tuple< typename Function::ResultType, typename Function::ResultType > result = evaluator(f, offspring.searchPoint());
		offspring.fitness(shark::tag::PenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >(result);
		offspring.fitness(shark::tag::UnpenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >(result);
		
		m_selection(m_pop);
		if (m_useNewUpdate) {
			if (offspring.selected()) {
				offspring.get<0>().m_noSuccessfulOffspring += 1.0;
				parent.get<0>().m_noSuccessfulOffspring += 1.0;
			}
		} else {
			if (offspring.selected() && offspring.rank() <= parent.rank() ) {
				offspring.get<0>().m_noSuccessfulOffspring += 1.0;
				parent.get<0>().m_noSuccessfulOffspring += 1.0;
			}
		}
		
		//update strategy parameter
		offspring.get<0>().update();
		parent.get<0>().update();

		if(offspring.selected()){
			std::sort(m_pop.begin(), m_pop.end(), Individual::RankOrdering);
		}
			
		SolutionSetType solutionSet;
		for (unsigned int i = 0; i < m_mu; i++) {
			m_pop[i].age()++;
			solutionSet.push_back(shark::makeResultSet(m_pop[i].searchPoint(), m_pop[i].fitness(shark::tag::UnpenalizedFitness()))) ;
		}
		return solutionSet;
	}
};
}

typedef TypeErasedMultiObjectiveOptimizer< VectorSpace<double>, detail::SteadyStateMOCMA< HypervolumeIndicator > > SteadyStateMOCMA;
typedef TypeErasedMultiObjectiveOptimizer< VectorSpace<double>, detail::SteadyStateMOCMA< LeastContributorApproximator< FastRng, HypervolumeCalculator > > > ApproximatedVolumeSteadyStateMOCMA;
typedef TypeErasedMultiObjectiveOptimizer< VectorSpace<double>, detail::SteadyStateMOCMA< AdditiveEpsilonIndicator > > EpsilonSteadyStateMOCMA;

}

#endif
