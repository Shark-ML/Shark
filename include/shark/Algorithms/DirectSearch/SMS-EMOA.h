/*!
 * 
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
#include <shark/Algorithms/DirectSearch/Indicators/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/BinaryTournamentSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/RankShareComparator.h>

// SMS-EMOA specific stuff
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>

#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>
#include <shark/Core/ResultSets.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

#include <boost/foreach.hpp>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

namespace shark { namespace detail {

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
	/**
	* \brief The individual type of the SMS-EMOA.
	*/
	typedef TypedIndividual<RealVector> Individual;

	/**
	* \brief The population type of the SMS-EMOA.
	*/
	typedef std::vector<Individual> Population;

	Population m_pop; ///< Population of size \f$\mu + 1\f$.
	unsigned int m_mu; ///< Population size \f$\mu\f$.
	bool m_useApproximatedHypervolume;///< Flag for deciding whether to use the approximated hypervolume.

	shark::moo::PenalizingEvaluator m_evaluator; ///< Evaluation operator.
	FastNonDominatedSort m_fastNonDominatedSort; ///< Operator that provides Deb's fast non-dominated sort. 
	IndicatorBasedSelection<HypervolumeIndicator> m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.
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
		archive & BOOST_SERIALIZATION_NVP( m_fastNonDominatedSort );
		archive & BOOST_SERIALIZATION_NVP( m_selection );
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
	*
	* \param [in] node The configuration tree node.
	*/
	void configure( const PropertyTree & node ) {
		init( 
			node.get( "Mu", this_type::DEFAULT_MU() ),
			node.get( "CrossoverProbability", this_type::DEFAULT_PC() ),
			node.get( "NC", this_type::DEFAULT_NC() ),
			node.get( "NM", this_type::DEFAULT_NM() )
		);
	}

	/**
	* \brief Initializes the algorithm.
	* \param [in] mu The population size. 
	* \param [in] pc Crossover probability, default value: 0.8.
	* \param [in] nc Parameter of the simulated binary crossover operator, default value: 20.0.
	* \param [in] nm Parameter of the mutation operator, default value: 20.0.
	*/
	void init( unsigned int mu = this_type::DEFAULT_MU(),
		double pc = this_type::DEFAULT_PC(),
		double nc = this_type::DEFAULT_NC(),
		double nm = this_type::DEFAULT_NM()
	) {
		m_mu = mu;
		m_crossoverProbability = pc;
		m_sbx.m_prob = 0.5;
		m_sbx.m_nc = nc;
		m_mutator.m_nm = nm;

		m_selection.setMu( m_mu );
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
		BOOST_FOREACH( Individual & ind, m_pop ) {
			ind.age()=0;
			f.proposeStartingPoint( ind.searchPoint() );
			boost::tuple< typename Function::ResultType, typename Function::ResultType > result = m_evaluator( f, *ind );
			ind.fitness( shark::tag::PenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >( result );
			ind.fitness( shark::tag::UnpenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >( result );
		}

		m_sbx.init(f);
		m_mutator.init(f);
	}

	/**
	* \brief Executes one iteration of the algorithm.
	* \tparam The type of the objective to iterate upon.
	* \param [in] f The function to iterate upon.
	* \returns The Pareto-set/-front approximation after the iteration.
	*/
	template<typename Function>
	SolutionSetType step( const Function & f ) {
		shark::BinaryTournamentSelection< ParetoDominanceComparator<FitnessExtractor> > selection;

		Population::iterator parent1 = selection( m_pop.begin(), m_pop.begin() + m_mu );
		Population::iterator parent2 = selection( m_pop.begin(), m_pop.begin() + m_mu );

		Individual mate1( *parent1 );
		Individual mate2( *parent2 );

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

		boost::tuple< typename Function::ResultType, typename Function::ResultType > result = m_evaluator( f, m_pop.back().searchPoint() );
		m_pop.back().fitness( shark::tag::PenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::PENALIZED_RESULT >( result );
		m_pop.back().fitness( shark::tag::UnpenalizedFitness() ) = boost::get< shark::moo::PenalizingEvaluator::UNPENALIZED_RESULT >( result );

		m_selection( m_pop );

		std::partition( m_pop.begin(), m_pop.end(), Individual::IsSelected );	
		SolutionSetType solutionSet;
		for( Population::iterator it = m_pop.begin(); it != m_pop.begin() + m_mu; ++it ) {
			it->age()++;
			solutionSet.push_back( shark::makeResultSet( it->searchPoint(), it->fitness( shark::tag::UnpenalizedFitness() ) ) ) ;
		}

		return solutionSet;
	}
};
}

typedef TypeErasedMultiObjectiveOptimizer< VectorSpace<double>, detail::SMSEMOA > SMSEMOA;
}


#endif
