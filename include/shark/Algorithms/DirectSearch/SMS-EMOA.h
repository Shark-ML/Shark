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

// MOO specific stuff
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/Indicators/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/TournamentSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>

#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>

#include <boost/foreach.hpp>

namespace shark {

/**
* \brief Implements the SMS-EMOA.
*
* Please see the following paper for further reference:
*	- Beume, Naujoks, Emmerich. 
*	SMS-EMOA: Multiobjective selection based on dominated hypervolume. 
*	European Journal of Operational Research.
*/
class SMSEMOA : public AbstractMultiObjectiveOptimizer<RealVector >{
private:
	/// \brief The individual type of the SMS-EMOA.
	typedef shark::Individual<RealVector,RealVector> Individual;
public:
	SMSEMOA() {
		mu() = 100;
		crossoverProbability() = 0.9;
		nc() = 20.0;
		nm() = 20.0;
	}

	std::string name() const {
		return "SMSEMOA";
	}
	
	/// \brief Returns the probability that crossover is applied.
	double crossoverProbability()const{
		return m_crossoverProbability;
	}
	/// \brief Returns the probability that crossover is applied.
	double& crossoverProbability(){
		return m_crossoverProbability;
	}
	
	double nm()const{
		return m_mutator.m_nm;
	}
	double& nm(){
		return m_mutator.m_nm;
	}
	
	double nc()const{
		return m_crossover.m_nc;
	}
	double& nc(){
		return m_crossover.m_nc;
	}
	
	unsigned int mu()const{
		return m_mu;
	}
	unsigned int& mu(){
		return m_mu;
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
		archive & BOOST_SERIALIZATION_NVP(m_mu);
		archive & BOOST_SERIALIZATION_NVP(m_best);

		archive & BOOST_SERIALIZATION_NVP( m_evaluator );
		archive & BOOST_SERIALIZATION_NVP( m_selection );
		archive & BOOST_SERIALIZATION_NVP( m_crossover );
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
		mu() = node.get( "Mu", mu() );
		crossoverProbability() = node.get( "CrossoverProbability", crossoverProbability() );
		nc() = node.get( "NC", nc() );
		nm() = node.get( "NM", nm() );
	}

	using AbstractMultiObjectiveOptimizer<RealVector >::init;
	/**
	 * \brief Initializes the algorithm for the supplied objective function.
	 * 
	 * \param [in] function The objective function.
	 * \param [in] startingPoints A set of intiial search points.
	 */
	void init( 
		ObjectiveFunctionType const& function, 
		std::vector<SearchPointType> const& startingPoints
	){
		m_pop.resize( mu() + 1 );
		m_best.resize(mu());
		for(std::size_t i = 0; i != mu(); ++i){
			m_pop[i].age()=0;
			function.proposeStartingPoint( m_pop[i].searchPoint() );
			m_evaluator( function, m_pop[i] );
			m_best[i].point = m_pop[i].searchPoint();
			m_best[i].value = m_pop[i].unpenalizedFitness();
		}
		m_selection( m_pop, m_mu );
		m_crossover.init(function);
		m_mutator.init(function);
	}

	/**
	 * \brief Executes one iteration of the algorithm.
	 * 
	 * \param [in] function The function to iterate upon.
	 */
	void step( ObjectiveFunctionType const& function ) {
		TournamentSelection< Individual::RankOrdering > selection;

		Individual mate1( *selection( m_pop.begin(), m_pop.begin() + mu() ) );
		Individual mate2( *selection( m_pop.begin(), m_pop.begin() + mu() ) );

		if( Rng::coinToss( m_crossoverProbability ) ) {
			m_crossover( mate1, mate2 );
		}

		if( Rng::coinToss() ) {
			m_mutator( mate1 );
			m_pop.back() = mate1;
		} else {					
			m_mutator( mate2 );
			m_pop.back() = mate2;
		}

		m_evaluator( function, m_pop.back() );
		m_selection( m_pop, m_mu );

		//if the individual got selected, insert it into the parent population
		if(m_pop.back().selected()){
			for(std::size_t i = 0; i != mu(); ++i){
				if(!m_pop[i].selected()){
					m_best[i].point = m_pop[mu()].searchPoint();
					m_best[i].value = m_pop[mu()].unpenalizedFitness();
					m_pop[i] = m_pop.back();
					break;
				}
			}
		}
	}
private:

	std::vector<Individual> m_pop; ///< Population of size \f$\mu + 1\f$.
	unsigned int m_mu; ///< Size of parent generation

	PenalizingEvaluator m_evaluator; ///< Evaluation operator.
	IndicatorBasedSelection<HypervolumeIndicator> m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.
	SimulatedBinaryCrossover< RealVector > m_crossover; ///< Crossover operator.
	PolynomialMutator m_mutator; ///< Mutation operator.

	double m_crossoverProbability; ///< Crossover probability.
};
}


#endif
