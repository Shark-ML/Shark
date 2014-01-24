/*!
 * 
 * \file        MOCMA.h
 *
 * \brief       Implements the generational Multi-objective Covariance Matrix Adapation ES.
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_MOCMA
#define SHARK_ALGORITHMS_DIRECT_SEARCH_MOCMA

#include <shark/Algorithms/DirectSearch/TypedIndividual.h>
#include <shark/Algorithms/DirectSearch/CMA/Chromosome.h>
#include <shark/Algorithms/DirectSearch/Operators/Initializers/CMA/Initializer.h>
#include <shark/Algorithms/DirectSearch/Operators/ParameterUpdate/CMA/Updater.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/CMA/Mutator.h>

// MOO specific stuff
#include <shark/Algorithms/DirectSearch/ParetoDominanceComparator.h>
#include <shark/Algorithms/DirectSearch/FastNonDominatedSort.h>
#include <shark/Algorithms/DirectSearch/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Indicators/AdditiveEpsilonIndicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/SteadyStateIndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/ApproximatedHypervolumeSelection.h>
#include <shark/Algorithms/DirectSearch/RankShareComparator.h>

#include <shark/Core/Traits/OptimizerTraits.h>

#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>
#include <shark/Core/ResultSets.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

#include <shark/Statistics/Statistics.h>

#include <boost/foreach.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/parameter.hpp>

#include <fstream>
#include <iterator>
#include <numeric>

namespace shark {

namespace mocma {

/**
 * \brief Chromosome type of the mocma.
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
typedef shark::elitist_cma::Initializer<
Individual,
Chromosome,
shark::ChromosomeIndex< 0 >::INDEX
> Initializer;

/**
 * \brief Strategy parameter updater.
 */
typedef shark::elitist_cma::Updater<
Individual,
Chromosome,
shark::ChromosomeIndex< 0 >::INDEX
> Updater;

/**
 * \brief Mutation operator.
 */
typedef shark::cma::Variator<
Individual,
Chromosome,
shark::ChromosomeIndex< 0 >::INDEX
> Variator;

/** \cond */
struct IndexComparator {
	IndexComparator(mocma::Population &pop) : m_pop(pop) {}

	bool operator()(unsigned int i, unsigned int j) {
		return(m_comp(m_pop[i], m_pop[j]));
	}

	RankShareComparator m_comp;
	mocma::Population &m_pop;
};
/** \endcond */
}

namespace detail {

/**
 * \brief Implements the generational MO-CMA-ES.
 *
 * Please see the following papers for further reference:
 *	- Igel, Suttorp and Hansen. Steady-state Selection and Efficient Covariance Matrix Update in the Multi-Objective CMA-ES.
 *	- Voﬂ, Hansen and Igel. Improved Step Size Adaptation for the MO-CMA-ES.
 */
template<typename Indicator = HypervolumeIndicator>
class MOCMA {
public:

	shark::mocma::Population m_pop; ///< Population of size \f$\mu+\mu\f$.
	mocma::Initializer m_initializer; ///< Initialization operator.
	mocma::Variator m_variator; ///< Mutation operator.
	mocma::Updater m_updater; ///< Strategy parameter update operator.

	shark::moo::PenalizingEvaluator m_evaluator; ///< Evaluation operator.

	RankShareComparator m_rsc; ///< Comparator for individuals based on their multi-objective rank and share.
	FastNonDominatedSort m_fastNonDominatedSort; ///< Operator that provides Deb's fast non-dominated sort.
	IndicatorBasedSelection< Indicator > m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.
	ApproximatedHypervolumeSelection m_approximatedSelection; ///< Selection operator relying on the approximated (contributing) hypervolume indicator.

	unsigned int m_mu; ///< Population size \f$\mu\f$.
	bool m_useNewUpdate; ///< Flag for deciding whether the improved step-size adaptation shall be used.
	bool m_useApproximatedHypervolume; ///< Flag for deciding whether to use the approximated hypervolume.
	bool m_useLogScaling; ///< Flag for deciding whether to use the logarithmic hypervolume indicator.

public:
	/**
	 * \brief The result type of the optimizer, a vector of tuples \f$( \vec x, \vec{f}( \vec{x} )\f$.
	 */
	typedef std::vector< ResultSet< RealVector, RealVector > > SolutionSetType;

	/**
	 * \brief Typedef of the algorithm's own type.
	 */
	typedef MOCMA<Indicator> this_type;

	/**
	 * \brief Default parent population size.
	 */
	static std::size_t DEFAULT_MU()	{
		return(100);
	}

	/**
	 * \brief Default penalty factor for penalizing infeasible solutions.
	 */
	static double DEFAULT_PENALTY_FACTOR() {
		return(1E-6);
	}

	/**
	 * \brief Default success threshold for step-size adaptation.
	 */
	static double DEFAULT_SUCCESS_THRESHOLD() {
		return(0.44);
	}

	/**
	 * \brief Default notion of success.
	 */
	static const char *DEFAULT_NOTION_OF_SUCCESS()	{
		return("IndividualBased");
	}

	/**
	 * \brief Default decision whether to rely on rescaled axis for hypervolume calculation.
	 */
	static bool DEFAULT_USE_LOG_HYP() {
		return(false);
	}

	/**
	 * \brief Default choice whether to rely on the approximated hypervolume indicator.
	 */
	static bool DEFAULT_USE_APPROXIMATED_HYPERVOLUME() {
		return(false);
	}

	/**
	 * \brief Default error bound for the approximated hypervolume indicator.
	 */
	static double DEFAULT_ERROR_BOUND()	{
		return(1E-2);
	}

	/**
	 * \brief Default error probability for the approximated hypervolume indicator.
	 */
	static double DEFAULT_ERROR_PROBABILITY() {
		return(1E-2);
	}

	/**
	 * \brief Default c'tor.
	 */
	MOCMA() {
		init();
	}

	/**
	 * \brief Returns the name of the algorithm.
	 */
	std::string name() const {
		return("MOCMA");
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
		archive &BOOST_SERIALIZATION_NVP(m_updater);

		archive &BOOST_SERIALIZATION_NVP(m_evaluator);
		archive &BOOST_SERIALIZATION_NVP(m_selection);
		archive &BOOST_SERIALIZATION_NVP(m_approximatedSelection);

		archive &BOOST_SERIALIZATION_NVP(m_mu);
		archive &BOOST_SERIALIZATION_NVP(m_useNewUpdate);
		archive &BOOST_SERIALIZATION_NVP(m_useApproximatedHypervolume);
		archive &BOOST_SERIALIZATION_NVP(m_useLogScaling);
	}

	/**
	 * \brief Initializes the algorithm.
	 * \param [in] mu The population size.
	 * \param [in] penaltyFactor The penalty factor for penalizing infeasible solutions.
	 * \param [in] successThreshold The success threshold \f$p_{\text{thresh}}\f$ for cutting off evolution path updates.
	 * \param [in] notionOfSuccess The notion of success.
	 * \param [in] useLogHyp Flag to determine whether to use the logarithmic hypervolume or not.
	 * \param [in] useApproximatedHypervolume Flag to determine whether to use the approximated hypervolume.
	 * \param [in] errorBound The error bound \f$ \epsilon \f$ for the approximated hypervolume selection.
	 * \param [in] errorProbability The error prob. \f$ \delta \f$ for the approximated hypervolume selection.
	 */
	void init(unsigned int mu = this_type::DEFAULT_MU(),
		double penaltyFactor = this_type::DEFAULT_PENALTY_FACTOR(),
	        double successThreshold = this_type::DEFAULT_SUCCESS_THRESHOLD(),
	        const std::string &notionOfSuccess = this_type::DEFAULT_NOTION_OF_SUCCESS(),
	        bool useLogHyp = this_type::DEFAULT_USE_LOG_HYP(),
	        bool useApproximatedHypervolume = this_type::DEFAULT_USE_APPROXIMATED_HYPERVOLUME(),
	        double errorBound = this_type::DEFAULT_ERROR_BOUND(),
	        double errorProbability = this_type::DEFAULT_ERROR_PROBABILITY()
	) {
		m_mu = mu;

		m_selection.setMu(mu);
		m_approximatedSelection.setMu(mu);

		m_evaluator.m_penaltyFactor = penaltyFactor;
		m_updater.m_successThreshold = successThreshold;

		if (notionOfSuccess == "IndividualBased") {
			m_useNewUpdate = false;
		} else if (notionOfSuccess == "PopulationBased") {
			m_useNewUpdate = true;
		}

		m_selection.m_useLogHyp = false/*useLogHyp*/;
		//m_selection.m_indicator.m_hv.m_useLogHyp = m_selection.m_useLogHyp;

		m_useApproximatedHypervolume = useApproximatedHypervolume;
		m_approximatedSelection.m_errorBound = errorBound;
		m_approximatedSelection.m_errorProbability = errorProbability;
	}

	/**
	 * \brief Initializes the algorithm from a configuration-tree node.
	 *
	 * The following sub keys are recognized:
	 *	- Mu, type: unsigned int, default value: 100.
	 *	- PenaltyFactor, type: double, default value: \f$10^{-6}\f$.
	 *	- SuccessThreshold, type: double, default value: 0.44.
	 *	- NotionOfSuccess, type: string, default value: IndividualBased.
	 *	- UseLogHyp, type: bool, default value: false.
	 *	- UseApproximatedHypervolume, type: bool, default value: false.
	 *	- ErrorBound, type: double, default value: \f$10^{-6}\f$.
	 *	- ErrorProbability, type: double, default value: \f$10^{-6}\f$.
	 *
	 * \param [in] node The configuration tree node.
	 */
	void configure(const PropertyTree &node) {
		init(
		    node.get<unsigned int>("Mu", this_type::DEFAULT_MU()),
		    node.get<double>("PenaltyFactor", this_type::DEFAULT_PENALTY_FACTOR()),
		    node.get<double>("SuccessThreshold", this_type::DEFAULT_SUCCESS_THRESHOLD()),
		    node.get<std::string>("NotionOfSuccess", this_type::DEFAULT_NOTION_OF_SUCCESS()),
		    node.get<bool>("UseLogHyp", this_type::DEFAULT_USE_LOG_HYP()),
		    node.get<bool>("UseApproximatedHypervolume", this_type::DEFAULT_USE_APPROXIMATED_HYPERVOLUME()),
		    node.get<double>("ErrorBound", this_type::DEFAULT_ERROR_BOUND()),
		    node.get<double>("ErrorProbability", this_type::DEFAULT_ERROR_PROBABILITY())
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
	void init(const ObjectiveFunction &f, const shark::RealVector &sp = shark::RealVector()) {

		RealVector ips[3];
		f.proposeStartingPoint(ips[0]);
		f.proposeStartingPoint(ips[1]);
		f.proposeStartingPoint(ips[2]);
		double d[3];
		d[0] = blas::norm_2(ips[1] - ips[0]);
		d[1] = blas::norm_2(ips[2] - ips[0]);
		d[2] = blas::norm_2(ips[2] - ips[1]);
		std::sort(d, d+3);

		m_initializer.m_initialSigma = d[1];
		m_pop.resize(2 * m_mu);

		std::size_t noObjectives = 0;

		shark::moo::PenalizingEvaluator evaluator;
		BOOST_FOREACH(mocma::Individual & ind, m_pop) {
			f.proposeStartingPoint(*ind);
			boost::tuple< typename ObjectiveFunction::ResultType, typename ObjectiveFunction::ResultType > 	result = evaluator(f, *ind);
			ind.fitness(shark::tag::PenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >(result);
			ind.fitness(shark::tag::UnpenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >(result);

			noObjectives = std::max(noObjectives, ind.fitness(shark::tag::PenalizedFitness()).size());
		}

		m_initializer.m_useNewUpdate = m_useNewUpdate;
		m_initializer.m_searchSpaceDimension = ips[0].size();
		m_initializer.m_noObjectives = noObjectives;

		BOOST_FOREACH(mocma::Individual & ind, m_pop) {
			m_initializer(ind);
		}

		m_selection.setNoObjectives(noObjectives);
		m_approximatedSelection.m_noObjectives = noObjectives;
	}

	/**
	 * \brief Executes one iteration of the algorithm.
	 * \tparam The type of the objective to iterate upon.
	 * \param [in] f The function to iterate upon.
	 * \returns The Pareto-set/-front approximation after the iteration.
	 */
	template<typename Function>
	SolutionSetType step(const Function &f) {

		static RankShareComparator rsc;

		mocma::Population::iterator itParents, itOffspring;
		for (itParents = m_pop.begin(), itOffspring = m_pop.begin() + m_mu;
		        itParents != m_pop.begin() + m_mu && itOffspring != m_pop.end();
		        ++itParents, ++itOffspring) {
			*itOffspring = *itParents;
			m_variator(*itOffspring);
			itOffspring->age() = 0;
			boost::tuple< typename Function::ResultType, typename Function::ResultType > 	result = m_evaluator(f, *(*itOffspring));
			itOffspring->fitness(shark::tag::PenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >(result);
			itOffspring->fitness(shark::tag::UnpenalizedFitness()) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >(result);

			itParents->get<0>().mep_parent = NULL;
			itOffspring->get<0>().mep_parent = &(itParents->get<0>());
		}

		m_fastNonDominatedSort(m_pop);

		if (m_useApproximatedHypervolume)
			m_approximatedSelection(m_pop);
		else
			m_selection(m_pop);

		if (m_useNewUpdate) {
			std::vector<unsigned int> indices(2 * m_mu);
			for (unsigned int i = 0; i < 2 * m_mu; i++) {
				indices[i] = i;
			}

			std::sort(
			    indices.begin(),
			    indices.end(),
			    //( rsc( boost::lambda::var( m_pop )[boost::lambda::_1], boost::lambda::var( m_pop )[boost::lambda::_2] ) )
			    mocma::IndexComparator(m_pop)
			);
			for (unsigned int i = 0; i < m_mu; i++) {
				if (indices[i] >= m_mu) {
					m_pop[indices[i]].template get<0>().m_noSuccessfulOffspring += 1.0;
					m_pop[indices[i]].template get<0>().mep_parent->m_noSuccessfulOffspring += 1.0;

				}
			}


		} else {

			for (unsigned int i = 0; i < m_mu; i++) {
				if (rsc(m_pop[m_mu + i], m_pop[i])) {
					m_pop[m_mu + i].template get<0>().m_noSuccessfulOffspring += 1.0;
					m_pop[i].template get<0>().m_noSuccessfulOffspring += 1.0;
				}
			}
		}

		SolutionSetType solutionSet;

		std::partial_sort(m_pop.begin(),
		        m_pop.begin() + m_mu,
		        m_pop.end(),
		        rsc
		);

		for (unsigned int i = 0; i < m_mu; i++) {
			m_pop[i].age()++;
			m_updater(m_pop[i]);

			solutionSet.push_back(shark::makeResultSet(*m_pop[i], m_pop[i].fitness(shark::tag::UnpenalizedFitness())));
		}

		return(solutionSet);
	};

};
}
/**
 * \brief \f$(\mu+1)\f$-MO-CMA-ES specialization of optimizer traits.
 */
template< typename Indicator >
struct OptimizerTraits< detail::MOCMA<Indicator> > {

	/**
	 * \brief Typedef of the algorithm type.
	 */
	typedef detail::MOCMA< Indicator > algorithm_type;

	/**
	 * \brief Prints out the configuration options and usage remarks of the algorithm to the supplied stream.
	 * \tparam Stream The type of the stream to output to.
	 * \param [in,out] s The stream to print usage information to.
	 */
	template<typename Stream>
	static void usage(Stream &s) {
		s << "MOCMA usage information:" << std::endl;
		s << "\t Mu, size of the population, default value: " << algorithm_type::DEFAULT_MU() << std::endl;
		s << "\t PenaltyFactor, factor for penalizing infeasible solutions, default value: " << algorithm_type::DEFAULT_PENALTY_FACTOR() << std::endl;
		s << "\t SuccessThreshold, success threshold for stalling strategy parameter updates, default value: " << algorithm_type::DEFAULT_SUCCESS_THRESHOLD << std::endl;
		s << "\t NotionOfSuccess, whether to carry out success estimation on a per-individual or on a population basis, default value: " << algorithm_type::DEFAULT_NOTION_OF_SUCCESS() << std::endl;
		s << "\t UseLogHyp, whether to rescale axis of the objective space logarithmically for hypervolume calculation, default value: " << algorithm_type::DEFAULT_USE_LOG_HYP() << std::endl;
		s << "\t UseApproximatedHypervolume, whether to use the exact or the approximated hypervolume, default value: " << algorithm_type::DEFAULT_USE_APPROXIMATED_HYPERVOLUME << std::endl;
		s << "\t ErrorBound, parameter epsilon of the hypervolume approximation scheme, default value: " << algorithm_type::DEFAULT_ERROR_BOUND() << std::endl;
		s << "\t ErrorProbability, parameter delta of the hypervolume approximation scheme, default value: " << algorithm_type::DEFAULT_ERROR_PROBABILITY() << std::endl;
	}

	/**
	 * \brief Assembles a default configuration for the algorithm.
	 * \tparam Tree structures modelling the boost::ptree concept.
	 * \param [in,out] node The tree to be filled with default key-value pairs.
	 */
	template<typename Tree>
	static void defaultConfig(Tree &node) {
		node.template add<unsigned int>(
		    "Mu",
		    algorithm_type::DEFAULT_MU()
		);
		node.template add<double>(
		    "PenaltyFactor",
		    algorithm_type::DEFAULT_PENALTY_FACTOR()
		);
		node.template add<double>(
		    "SuccessThreshold",
		    algorithm_type::DEFAULT_SUCCESS_THRESHOLD()
		);
		node.template add<std::string>(
		    "NotionOfSuccess",
		    algorithm_type::DEFAULT_NOTION_OF_SUCCESS()
		);
		node.template add<bool>(
		    "UseLogHyp",
		    algorithm_type::DEFAULT_USE_LOG_HYP()
		);
		node.template add<bool>(
		    "UseApproximatedHypervolume",
		    algorithm_type::DEFAULT_USE_APPROXIMATED_HYPERVOLUME()
		);
		node.template add<double>(
		    "ErrorBound",
		    algorithm_type::DEFAULT_ERROR_BOUND()
		);
		node.template add<double>(
		    "ErrorProbability",
		    algorithm_type::DEFAULT_ERROR_PROBABILITY()
		);
	}
};

/** \brief Injects the S-MOCMA into the inheritance hierarchy. */
typedef TypeErasedMultiObjectiveOptimizer< VectorSpace<double>, detail::MOCMA< HypervolumeIndicator > > MOCMA;

/** \brief Injects the Epsilon-MOCMA into the inheritance hierarchy. */
typedef TypeErasedMultiObjectiveOptimizer< VectorSpace<double>, detail::MOCMA< AdditiveEpsilonIndicator > > EpsilonMOCMA;

/** \brief Registers the MOCMA relying on the hypervolume indicator with the factory. */
ANNOUNCE_MULTI_OBJECTIVE_OPTIMIZER(MOCMA, moo::RealValuedMultiObjectiveOptimizerFactory)

/** \brief Registers the MOCMA relying on the additive epsilon indicator with the factory. */
ANNOUNCE_MULTI_OBJECTIVE_OPTIMIZER(EpsilonMOCMA, moo::RealValuedMultiObjectiveOptimizerFactory)

}

#endif
