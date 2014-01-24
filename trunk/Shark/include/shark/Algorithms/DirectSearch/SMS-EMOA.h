/*!
 * 
 * \file        SMS-EMOA.h
 *
 * \brief       Implements the SMS-EMOA.
 * 
 * See Nicola Beume, Boris Naujoks, and Michael Emmerich. 
 * SMS-EMOA: Multiobjective selection based on dominated hypervolume. 
 * European Journal of Operational Research, 181(3):1653-1669, 2007. 
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_SMS_EMOA_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_SMS_EMOA_H

#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>

#include <shark/Algorithms/DirectSearch/TypedIndividual.h>

// MOO specific stuff
#include <shark/Algorithms/DirectSearch/ParetoDominanceComparator.h>
#include <shark/Algorithms/DirectSearch/FastNonDominatedSort.h>
#include <shark/Algorithms/DirectSearch/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/BinaryTournamentSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/ApproximatedHypervolumeSelection.h>
#include <shark/Algorithms/DirectSearch/RankShareComparator.h>

// SMS-EMOA specific stuff
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>

#include <shark/Core/Traits/OptimizerTraits.h>

#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>
#include <shark/Core/ResultSets.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

#include <boost/foreach.hpp>
#include <boost/parameter.hpp>

#include <fstream>
#include <iterator>
#include <numeric>

#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

namespace shark { namespace detail {

namespace smsemoa {
	/**
	* \namespace Internal namespace of the SMS-EMOA.
	*/

	/**
	* \brief The individual type of the SMS-EMOA.
	*/
	typedef TypedIndividual<RealVector> Individual;

	/**
	* \brief The population type of the SMS-EMOA.
	*/
	typedef std::vector<Individual> Population;
}

/**
* \brief Implements the SMS-EMOA.
*
* Please see the following paper for further reference:
*	- Beume, Naujoks, Emmerich. 
*	SMS-EMOA: Multiobjective selection based on dominated hypervolume. 
*	European Journal of Operational Research.
*/
class SMSEMOA {
protected:
	smsemoa::Population m_pop; ///< Population of size \f$\mu + 1\f$.
	unsigned int m_mu; ///< Population size \f$\mu\f$.
	bool m_useApproximatedHypervolume;///< Flag for deciding whether to use the approximated hypervolume.

	shark::moo::PenalizingEvaluator m_evaluator; ///< Evaluation operator.
	RankShareComparator rsc; ///< Comparator for individuals based on their multi-objective rank and share.
	FastNonDominatedSort m_fastNonDominatedSort; ///< Operator that provides Deb's fast non-dominated sort. 
	IndicatorBasedSelection<HypervolumeIndicator> m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.
	ApproximatedHypervolumeSelection m_approximatedSelection; ///< Selection operator relying on the approximated (contributing) hypervolume indicator.
	SimulatedBinaryCrossover< RealVector > m_sbx; ///< Crossover operator.
	PolynomialMutator m_mutator; ///< Mutation operator.

	double m_crossoverProbability; ///< Crossover probability.

public:
	/**
	* \brief The result type of the optimizer, a vector of tuples \f$( \vec x, \vec{f}( \vec{x} )\f$.
	*/
	typedef std::vector< ResultSet< RealVector, RealVector > > SolutionSetType;

	typedef SMSEMOA this_type;

	/**
	* \brief Default parent population size.
	*/
	static unsigned int DEFAULT_MU() {
		return( 100 );
	}

	/**
	* \brief Default crossover probability.
	*/
	static double DEFAULT_PC() {
		return( 0.9 );
	}

	/**
	* \brief Default value for nc.
	*/
	static double DEFAULT_NC() {
		return( 20.0 );
	}

	/**
	* \brief Default value for nm.
	*/
	static double DEFAULT_NM() {
		return( 20.0 );
	}

	/**
	* \brief Default choice whether to rely on the approximated hypervolume indicator.
	*/
	static bool DEFAULT_USE_APPROXIMATED_HYPERVOLUME() {
		return( false );
	}

	/**
	* \brief Default error bound for the approximated hypervolume indicator.
	*/
	static double DEFAULT_ERROR_BOUND() {
		return( 1E-2 );
	}

	/**
	* \brief Default error probability for the approximated hypervolume indicator.
	*/
	static double DEFAULT_ERROR_PROBABILITY() {
		return( 1E-2 );
	}

	/**
	* \brief Default c'tor.
	*/
	SMSEMOA() {
		init();
	}

	/**
	* \brief Returns the name of the algorithm.
	*/
	std::string name() const {
		return( "SMSEMOA" );
	}

	/**
	* \brief Stores/loads the algorithm's state.
	* \tparam Archive The type of the archive.
	* \param [in,out] archive The archive to use for loading/storing.
	* \param [in] version Currently unused.
	*/
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
		archive & BOOST_SERIALIZATION_NVP( m_pop );
		archive & BOOST_SERIALIZATION_NVP( m_mu );

		archive & BOOST_SERIALIZATION_NVP( m_evaluator );
		archive & BOOST_SERIALIZATION_NVP( rsc );
		archive & BOOST_SERIALIZATION_NVP( m_fastNonDominatedSort );
		archive & BOOST_SERIALIZATION_NVP( m_selection );
		archive & BOOST_SERIALIZATION_NVP( m_approximatedSelection );
		archive & BOOST_SERIALIZATION_NVP( m_sbx );
		archive & BOOST_SERIALIZATION_NVP( m_mutator );

		archive & BOOST_SERIALIZATION_NVP( m_crossoverProbability );
	}

	/**
	* \brief Initializes the algorithm from a configuration-tree node.
	*
	* The following sub keys are recognized:
	*	- Mu, type: unsigned int, default value: 100.
	*	- CrossoverProbability, type: double, default value: 0.9.
	*	- NC, type: double, default value: 20.
	*	- NM, type: double; default value: 20.
	*	- UseApproximatedHypervolume, type: bool, default value: false.
	*	- ErrorBound, type: double, default value: \f$10^{-6}\f$.
	*	- ErrorProbability, type: double, default value: \f$10^{-6}\f$.
	*
	* \param [in] node The configuration tree node.
	*/
	void configure( const PropertyTree & node ) {
		init( 
			node.get( "Mu", this_type::DEFAULT_MU() ),
			node.get( "CrossoverProbability", this_type::DEFAULT_PC() ),
			node.get( "NC", this_type::DEFAULT_NC() ),
			node.get( "NM", this_type::DEFAULT_NM() ),
			node.get( "UseApproximatedHypervolume", this_type::DEFAULT_USE_APPROXIMATED_HYPERVOLUME() ),
			node.get( "ErrorBound", this_type::DEFAULT_ERROR_BOUND() ),
			node.get( "ErrorProbability", this_type::DEFAULT_ERROR_PROBABILITY() )
			);
	}

	/**
	* \brief Initializes the algorithm.
	* \param [in] mu The population size. 
	* \param [in] pc Crossover probability, default value: 0.8.
	* \param [in] nc Parameter of the simulated binary crossover operator, default value: 20.0.
	* \param [in] nm Parameter of the mutation operator, default value: 20.0.
	* \param [in] useApproximatedHypervolume Flag to determine whether to use the approximated hypervolume.
	* \param [in] errorBound The error bound \f$ \epsilon \f$ for the approximated hypervolume selection.
	* \param [in] errorProbability The error prob. \f$ \delta \f$ for the approximated hypervolume selection.
	*/
	void init( unsigned int mu = this_type::DEFAULT_MU(),
		double pc = this_type::DEFAULT_PC(),
		double nc = this_type::DEFAULT_NC(),
		double nm = this_type::DEFAULT_NM(),
		bool useApproximatedHypervolume = this_type::DEFAULT_USE_APPROXIMATED_HYPERVOLUME(),
		double errorBound = this_type::DEFAULT_ERROR_BOUND(),
		double errorProbability = this_type::DEFAULT_ERROR_PROBABILITY() 
	) {
		m_mu = mu;
		m_crossoverProbability = pc;
		m_sbx.m_prob = 0.5;
		m_sbx.m_nc = nc;
		m_mutator.m_nm = nm;

		m_selection.setMu( m_mu );
		m_approximatedSelection.setMu( m_mu );

		m_useApproximatedHypervolume = useApproximatedHypervolume;
		m_approximatedSelection.m_errorBound = errorBound;
		m_approximatedSelection.m_errorProbability = errorProbability;
	}

	/**
	* \brief Initializes the algorithm for the supplied objective function.
	* \tparam ObjectiveFunction The type of the objective function, 
	* needs to adhere to the concept of an AbstractObjectiveFunction.
	* \param [in] f The objective function.
	* \param [in] sp An initial search point.
	*/
	template<typename Function>
	void init( const Function & f, const RealVector & sp ) {
		m_pop.resize( m_mu + 1 );

		std::size_t noObjectives = 0;

		BOOST_FOREACH( smsemoa::Individual & ind, m_pop ) {
			ind.age()=0;

			f.proposeStartingPoint( *ind );
			boost::tuple< typename Function::ResultType, typename Function::ResultType > result = m_evaluator( f, *ind );
			ind.fitness( shark::tag::PenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >( result );
			ind.fitness( shark::tag::UnpenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >( result );

			ind.setNoObjectives( ind.fitness( shark::tag::PenalizedFitness() ).size() );
			noObjectives = std::max( noObjectives, ind.fitness( shark::tag::PenalizedFitness() ).size() );
		}

		m_selection.setNoObjectives( noObjectives );
		m_approximatedSelection.setNoObjectives( noObjectives );

		m_sbx.init(f);
		m_mutator.init(f);

		m_evaluator.m_penaltyFactor = 1E-6;
	}

	/**
	* \brief Executes one iteration of the algorithm.
	* \tparam The type of the objective to iterate upon.
	* \param [in] f The function to iterate upon.
	* \returns The Pareto-set/-front approximation after the iteration.
	*/
	template<typename Function>
	SolutionSetType step( const Function & f ) {
		static shark::ParetoDominanceComparator< shark::tag::PenalizedFitness > pdc;
		static shark::BinaryTournamentSelection< 
			shark::ParetoDominanceComparator< 
				shark::tag::PenalizedFitness 
			> 
		> selection( pdc );

		smsemoa::Population::iterator parent1 = selection( m_pop.begin(), m_pop.begin() + m_mu );
		smsemoa::Population::iterator parent2 = selection( m_pop.begin(), m_pop.begin() + m_mu );

		smsemoa::Individual mate1( *parent1 );
		smsemoa::Individual mate2( *parent2 );

		if( Rng::coinToss( m_crossoverProbability ) ) {
			m_sbx( mate1, mate2 );
		}

		if( Rng::coinToss() ) {
			m_mutator( mate1 );
			m_pop.back() = mate1;
		} else {					
			m_mutator( mate2 );
			m_pop.back() = mate2;
		}

		boost::tuple< typename Function::ResultType, typename Function::ResultType > result = m_evaluator( f, *m_pop.back() );
		m_pop.back().fitness( shark::tag::PenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >( result );
		m_pop.back().fitness( shark::tag::UnpenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >( result );

		m_fastNonDominatedSort( m_pop );
		if( m_useApproximatedHypervolume )
			m_approximatedSelection( m_pop );
		else
			m_selection( m_pop );

		std::sort( m_pop.begin(), m_pop.end(), RankShareComparator() );	
		SolutionSetType solutionSet;
		for( smsemoa::Population::iterator it = m_pop.begin(); it != m_pop.begin() + m_mu; ++it ) {
			it->age()++;
			solutionSet.push_back( shark::makeResultSet( *(*it), it->fitness( shark::tag::UnpenalizedFitness() ) ) ) ;
		}

		return( solutionSet );
	}
};
}

/**
* \brief SMS-EMOA specialization of optimizer traits.
*/
template<>
struct OptimizerTraits<detail::SMSEMOA> {

	typedef detail::SMSEMOA this_type;

	/**
	* \brief Prints out the configuration options and usage remarks of the algorithm to the supplied stream.
	* \tparam Stream The type of the stream to output to.
	* \param [in,out] s The stream to print usage information to.
	*/
	template<typename Stream>
	static void usage( Stream & s ) {
		s << "SMS-EMOA usage information:" << std::endl;
		s << "\t Mu, size of the population, default value: \t\t" << this_type::DEFAULT_MU() << std::endl;
		s << "\t CrossoverProbability, type: double, default value: \t\t" << this_type::DEFAULT_PC() << std::endl;
		s << "\t NC, type: double, default value: \t\t" << this_type::DEFAULT_NC() << std::endl;
		s << "\t NM, type: double; default value: \t\t" << this_type::DEFAULT_NM() << std::endl;
		s << "\t UseApproximatedHypervolume, whether to use the exact or the approximated hypervolume, default value: \t\t" << this_type::DEFAULT_USE_APPROXIMATED_HYPERVOLUME() << std::endl;
		s << "\t ErrorBound, parameter epsilon of the hypervolume approximation scheme, default value: \t\t" << this_type::DEFAULT_ERROR_BOUND() << std::endl;
		s << "\t ErrorProbability, parameter delta of the hypervolume approximation scheme, default value: \t\t" << this_type::DEFAULT_ERROR_PROBABILITY() << std::endl;
	}

	/**
	* \brief Assembles a default configuration for the algorithm.
	* \tparam Tree structures modelling the boost::ptree concept.
	* \param [in,out] node The tree to be filled with default key-value pairs.
	*/
	template<typename Tree>
	static void defaultConfig( Tree & node ) {
		node.template add<unsigned int>( "Mu", this_type::DEFAULT_MU() );
		node.template add<double>( "CrossoverProbability", this_type::DEFAULT_PC() );
		node.template add<double>( "NC", this_type::DEFAULT_NC() );
		node.template add<double>( "NM", this_type::DEFAULT_NM() );
		node.template add<bool>( "UseApproximatedHypervolume", this_type::DEFAULT_USE_APPROXIMATED_HYPERVOLUME() );
		node.template add<double>( "ErrorBound", this_type::DEFAULT_ERROR_BOUND() );
		node.template add<double>( "ErrorProbability", this_type::DEFAULT_ERROR_PROBABILITY() );
	}
};

/** \brief Injects the SMS-EMOA into the inheritance hierarchy. */
typedef TypeErasedMultiObjectiveOptimizer< VectorSpace<double>, detail::SMSEMOA > SMSEMOA;

ANNOUNCE_MULTI_OBJECTIVE_OPTIMIZER( SMSEMOA, moo::RealValuedMultiObjectiveOptimizerFactory )
}


#endif
