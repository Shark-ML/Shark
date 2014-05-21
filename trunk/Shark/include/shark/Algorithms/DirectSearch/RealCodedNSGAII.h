/*!
 * 
 *
 * \brief       IndicatorBasedRealCodedNSGAII.h
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

#include <shark/Algorithms/DirectSearch/Individual.h>

// MOO specific stuff
#include <shark/Algorithms/DirectSearch/Indicators/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Indicators/AdditiveEpsilonIndicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/TournamentSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>


namespace shark {

/**
* \brief Implements the NSGA-II.
*
* Please see the following papers for further reference:
*  Deb, Agrawal, Pratap and Meyarivan. 
*  A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II 
*  IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 6, NO. 2, APRIL 2002
*/
template<typename Indicator>
class IndicatorBasedRealCodedNSGAII : public AbstractMultiObjectiveOptimizer<RealVector >{
private:
	/**
	* \brief The individual type of the NSGA-II.
	*/
	typedef shark::Individual<RealVector,RealVector> Individual;

	std::vector<Individual> m_pop; ///< Population of size \f$\mu + 1\f$.
	unsigned int m_mu; ///< Size of parent generation

	IndicatorBasedSelection<Indicator> m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.

	PenalizingEvaluator m_evaluator; ///< Evaluation operator. 
	SimulatedBinaryCrossover< RealVector > m_crossover; ///< Crossover operator.
	PolynomialMutator m_mutator; ///< Mutation operator.

	double m_crossoverProbability; ///< Crossover probability.			
public:

	/**
	* \brief Default c'tor.
	*/
	IndicatorBasedRealCodedNSGAII(){
		mu() = 100;
		crossoverProbability() = 0.8;
		nc() = 20.0;
		nm() = 20.0;
	}

	std::string name() const {
		return "RealCodedNSGAII";
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
		archive & m_pop;
		archive & m_mu;
		archive & m_best;

		archive & m_evaluator;
		archive & m_crossover;
		archive & m_mutator;

		archive & m_crossoverProbability;
	}

	/**
	* \brief Initializes the algorithm from a configuration-tree node.
	*
	* The following sub keys are recognized:
	*	- Mu, type: unsigned int, default value: 100.
	*	- CrossoverProbability, type: double, default value: 0.8.
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
	* \tparam ObjectiveFunction The type of the objective function, 
	* needs to adhere to the concept of an AbstractObjectiveFunction.
	* \param [in] function The objective function
	* \param [in] startingPoints Starting point to initialize the algorithm for.
	*/
	void init( 
		ObjectiveFunctionType const& function, 
		std::vector<SearchPointType> const& startingPoints
	){
		//create parent set
		m_pop.reserve( 2 * mu() );
		m_pop.resize(mu());
		m_best.resize(mu());
		for(std::size_t i = 0; i != mu(); ++i){
			function.proposeStartingPoint( m_pop[i].searchPoint() );
		}
		//evaluate initial parent set and create best front
		m_evaluator( function, m_pop.begin(),m_pop.begin()+mu() );
		m_selection( m_pop,m_mu );
		for(std::size_t i = 0; i != mu(); ++i){
			m_best[i].point = m_pop[i].searchPoint();
			m_best[i].value = m_pop[i].unpenalizedFitness();
		}
		//make room for offspring
		m_pop.resize(2*mu());
		
		m_crossover.init(function);
		m_mutator.init(function);
	}

	/**
	 * \brief Executes one iteration of the algorithm.
	 * 
	 * \param [in] function The function to iterate upon.
	 */
	void step( ObjectiveFunctionType const& function ) {
		TournamentSelection< Individual::RankOrdering > matingSelection;
		
		matingSelection(
			m_pop.begin(), 
			m_pop.begin() + mu(),
			m_pop.begin() + mu(),
			m_pop.end()
		);

		for( unsigned int i = 1; i < mu(); i++ ) {
			if( Rng::coinToss( 0.8 ) ) {
				m_crossover( m_pop[mu() + i - 1], m_pop[mu() + i] );
			}
		}
		for( unsigned int i = 0; i < mu(); i++ ) {
			m_mutator( m_pop[mu() + i] );
		}
		m_evaluator( function, m_pop.begin()+mu(), m_pop.end() );
		m_selection( m_pop, m_mu );

		std::partition( m_pop.begin(), m_pop.end(), Individual::IsSelected );	

		for( std::size_t i = 0; i != mu(); ++i ) {
			noalias(m_best[i].value) = m_pop[i].unpenalizedFitness();
			noalias(m_best[i].point) = m_pop[i].searchPoint();
		}
	}
};

typedef IndicatorBasedRealCodedNSGAII< HypervolumeIndicator > RealCodedNSGAII;
typedef IndicatorBasedRealCodedNSGAII< AdditiveEpsilonIndicator > EpsRealCodedNSGAII;
}
#endif
