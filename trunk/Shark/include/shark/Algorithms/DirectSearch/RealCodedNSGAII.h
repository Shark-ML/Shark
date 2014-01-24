/*!
 * 
 * \file        RealCodedNSGAII.h
 *
 * \brief       RealCodedNSGAII.h
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_REAL_CODED_NSGA_II_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_REAL_CODED_NSGA_II_H

#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>

#include <shark/Algorithms/DirectSearch/TypedIndividual.h>

// MOO specific stuff
#include <shark/Algorithms/DirectSearch/ParetoDominanceComparator.h>
#include <shark/Algorithms/DirectSearch/FastNonDominatedSort.h>
#include <shark/Algorithms/DirectSearch/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/RankShareComparator.h>

// SMS-EMOA specific stuff
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/BinaryTournamentSelection.h>

#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>


namespace shark { namespace detail { 
	
namespace nsga2 {
	/**
	* \namespace Internal namespace of the NSGA-II.
	*/

	/**
	* \brief The individual type of the NSGA-II.
	*/
	typedef TypedIndividual<RealVector> Individual;

	/**
	* \brief The population type of the NSGA-II.
	*/
	typedef std::vector<Individual> Population;
}

/**
* \brief Implements the NSGA-II.
*
* Please see the following papers for further reference:
*  Deb, Agrawal, Pratap and Meyarivan. 
*  A Fast Elitist Non-dominated Sorting Genetic Algorithm for Multi-objective Optimization: NSGA-II. 
*  PPSN VI. 
*/
template<typename Indicator = shark::HypervolumeIndicator>
class RealCodedNSGAII {
protected:
	nsga2::Population m_pop; ///< Population of size \f$\mu + 1\f$.
	unsigned int m_mu; ///< Population size \f$\mu\f$.

	shark::moo::PenalizingEvaluator m_evaluator; ///< Evaluation operator.

	RankShareComparator m_rsc; ///< Comparator for individuals based on their multi-objective rank and share.
	FastNonDominatedSort m_fastNonDominatedSort; ///< Operator that provides Deb's fast non-dominated sort. 
	IndicatorBasedSelection<Indicator> m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.

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
	RealCodedNSGAII() : m_binaryTournamentSelection( m_rsc ) {
		init();
	}

	/**
	* \brief Accesses the name of the optimizer.
	*/
	std::string name() const {
		return( "RealCodedNSGAII" );
	}

	/**
	* \brief Stores/loads the algorithm's state.
	* \tparam Archive The type of the archive.
	* \param [in,out] archive The archive to use for loading/storing.
	* \param [in] version Currently unused.
	*/
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
		archive & m_pop; ///< Population of size \f$\mu + 1\f$.
		archive & m_mu; /// Population size \f$\mu\f$.

		archive & m_evaluator; ///< Evaluation operator.
		archive & m_rsc; ///< Comparator for individuals based on their multi-objective rank and share.
		archive & m_fastNonDominatedSort; ///< Operator that provides Deb's fast non-dominated sort. 
		archive & m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.
		archive & m_sbx; ///< Crossover operator.
		archive & m_mutator; ///< Mutation operator.

		archive & m_crossoverProbability; ///< Crossover probability.
	}

	/**
	* \brief Initializes the algorithm from a configuration-tree node.
	*
	* The following sub keys are recognized:
	*	- Mu, type: unsigned int, default value: 100.
	*	- CrossoverProbability, type: double, default value: 0.8.
	*	- NC, type: double, default value: 10.
	*	- NM, type: double; default value: 20.
	*
	* \param [in] node The configuration tree node.
	*/
	void configure( const PropertyTree & node ) {
		init( node.get( "Mu", 100 ),
			node.get( "CrossoverProbability", 0.8 ),
			node.get( "NC", 20.0 ),
			node.get( "NM", 20.0 )
			);
	}

	/**
	* \brief Initializes the algorithm.
	* \param [in] mu The population size. 
	* \param [in] pc Crossover probability, default value: 0.8.
	* \param [in] nc Parameter of the simulated binary crossover operator, default value: 10.0.
	* \param [in] nm Parameter of the mutation operator, default value: 20.0.
	*/
	void init( unsigned int mu = 100,
		double pc = 0.8,
		double nc = 20.0,
		double nm = 20.0
		) {
			m_mu = mu;
			m_crossoverProbability = pc;
			m_sbx.m_nc = nc;
			m_mutator.m_nm = nm;

			m_selection.setMu( m_mu );
	}

	/**
	* \brief Initializes the algorithm for the supplied objective function.
	* \tparam ObjectiveFunction The type of the objective function, 
	* needs to adhere to the concept of an AbstractObjectiveFunction.
	* \param [in] f The objective function
	* \param [in] sp Starting point to initialize the algorithm for.
	*/
	template<typename Function>
	void init( const Function & f, const RealVector & sp  ) {
		m_pop.resize( 2 * m_mu );

		std::size_t noObjectives = 0;

		BOOST_FOREACH( nsga2::Individual & ind, m_pop ) {
			ind.age()=0;

			f.proposeStartingPoint( *ind );
			boost::tuple< typename Function::ResultType, typename Function::ResultType > result = m_evaluator( f, *ind );
			ind.fitness( shark::tag::PenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >( result );
			ind.fitness( shark::tag::UnpenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >( result );

			ind.setNoObjectives( ind.fitness( shark::tag::PenalizedFitness() ).size() );
			noObjectives = std::max( noObjectives, ind.fitness( shark::tag::PenalizedFitness() ).size() );
		}
		
		m_selection.setNoObjectives( noObjectives );
		m_sbx.init(f);
		m_mutator.init(f);

		m_fastNonDominatedSort( m_pop );
		m_selection( m_pop );
	}

	/**
	* \brief Executes one iteration of the algorithm.
	* \tparam The type of the objective to iterate upon.
	* \param [in] f The function to iterate upon.
	* \returns The Pareto-set/-front approximation after the iteration.
	*/
	template<typename Function>
	SolutionSetType step( const Function & f ) {
		m_binaryTournamentSelection( m_pop.begin(), 
			m_pop.begin() + m_mu,
			m_pop.begin() + m_mu,
			m_pop.end() );

		for( unsigned int i = 1; i < m_mu; i++ ) {
			if( Rng::coinToss( 0.8 ) ) {
				m_sbx( m_pop[m_mu + i - 1], m_pop[m_mu + i] );
			}
		}

		for( unsigned int i = 0; i < m_mu; i++ ) {
			m_mutator( m_pop[m_mu + i] );
			m_pop[m_mu + i].age() = 0;
			boost::tuple< typename Function::ResultType, typename Function::ResultType > result = m_evaluator( f, *m_pop[ m_mu + i ] );
			m_pop[m_mu + i].fitness( shark::tag::PenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >( result );
			m_pop[m_mu + i].fitness( shark::tag::UnpenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >( result );

		}

		m_fastNonDominatedSort( m_pop );
		m_selection( m_pop );

		std::partial_sort( 
			m_pop.begin(), 
			m_pop.begin() + m_mu,
			m_pop.end(), 
			RankShareComparator()
			);	

		SolutionSetType solutionSet;
		for( nsga2::Population::iterator it = m_pop.begin(); it != m_pop.begin() + m_mu; ++it ) {
			it->age()++;
			solutionSet.push_back( shark::makeResultSet( *(*it), it->fitness( shark::tag::UnpenalizedFitness() ) ) ) ;
		}

		return( solutionSet );
	}
};
}

/**
* \brief NSGA-II specialization of optimizer traits.
*/
template<>
struct OptimizerTraits< detail::RealCodedNSGAII<> > {
	/**
	* \brief Prints out the configuration options and usage remarks of the algorithm to the supplied stream.
	* \tparam Stream The type of the stream to output to.
	* \param [in,out] s The stream to print usage information to.
	*/
	template<typename Stream>
	static void usage( Stream & s ) {
		s << "RealCodedNSGAII usage information:" << std::endl;
		s << "\t Mu, size of the population, default value: \t\t 100" << std::endl;
		s << "\t CrossoverProbability, type: double, default value: \t\t 0.8." << std::endl;
		s << "\t NC, type: double, default value: \t\t 10." << std::endl;
		s << "\t NM, type: double; default value: \t\t 20." << std::endl;
	}
};

/** \brief Injects the NSGA-II into the inheritance hierarchy. */
typedef TypeErasedMultiObjectiveOptimizer< VectorSpace<double>, detail::RealCodedNSGAII<> > HypRealCodedNSGAII;

/** \brief Injects the NSGA-II into the inheritance hierarchy. */
typedef TypeErasedMultiObjectiveOptimizer< VectorSpace<double>, detail::RealCodedNSGAII< shark::AdditiveEpsilonIndicator > > EpsRealCodedNSGAII;

/** \brief Registers the real-coded NSGA-II relying on the hypervolume indicator with the factory. */
ANNOUNCE_MULTI_OBJECTIVE_OPTIMIZER( HypRealCodedNSGAII, moo::RealValuedMultiObjectiveOptimizerFactory );

/** \brief Registers the real-coded NSGA-II relying on the additive epsilon indicator with the factory. */
ANNOUNCE_MULTI_OBJECTIVE_OPTIMIZER( EpsRealCodedNSGAII, moo::RealValuedMultiObjectiveOptimizerFactory )
}

#endif
