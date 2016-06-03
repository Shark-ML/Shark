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
public:
	SMSEMOA(DefaultRngType& rng = Rng::globalRng):mpe_rng(&rng) {
		m_mu = 100;
		m_mutator.m_nm = 20.0;
		m_crossover.m_nc = 20.0;
		m_crossoverProbability = 0.9;
		this->m_features |= AbstractMultiObjectiveOptimizer<RealVector >::CAN_SOLVE_CONSTRAINED;
	}

	std::string name() const {
		return "SMSEMOA";
	}
	
	/// \brief Returns the probability that crossover is applied.
	double crossoverProbability()const{
		return m_crossoverProbability;
	}
	
	double nm()const{
		return m_mutator.m_nm;
	}
	
	double nc()const{
		return m_crossover.m_nc;
	}
	
	unsigned int mu()const{
		return m_mu;
	}
	
	unsigned int& mu(){
		return m_mu;
	}
	
	HypervolumeIndicator& indicator(){
		return m_selection.indicator();
	}
	HypervolumeIndicator const& indicator()const{
		return m_selection.indicator();
	}

	void read( InArchive & archive ){
		archive & BOOST_SERIALIZATION_NVP( m_parents );
		archive & BOOST_SERIALIZATION_NVP( m_mu );
		archive & BOOST_SERIALIZATION_NVP( m_best );

		archive & BOOST_SERIALIZATION_NVP( m_selection );
		archive & BOOST_SERIALIZATION_NVP( m_crossover );
		archive & BOOST_SERIALIZATION_NVP( m_mutator );
		archive & BOOST_SERIALIZATION_NVP( m_crossoverProbability );
	}
	void write( OutArchive & archive ) const{
		archive & BOOST_SERIALIZATION_NVP( m_parents );
		archive & BOOST_SERIALIZATION_NVP( m_mu );
		archive & BOOST_SERIALIZATION_NVP( m_best );

		archive & BOOST_SERIALIZATION_NVP( m_selection );
		archive & BOOST_SERIALIZATION_NVP( m_crossover );
		archive & BOOST_SERIALIZATION_NVP( m_mutator );
		archive & BOOST_SERIALIZATION_NVP( m_crossoverProbability );
	}

	using AbstractMultiObjectiveOptimizer<RealVector >::init;
	/**
	 * \brief Initializes the algorithm for the supplied objective function.
	 * 
	 * \param [in] function The objective function.
	 * \param [in] initialSearchPoints A set of intiial search points.
	 */
	void init( 
		ObjectiveFunctionType& function, 
		std::vector<SearchPointType> const& initialSearchPoints
	){
		checkFeatures(function);
		std::vector<RealVector> values(initialSearchPoints.size());
		for(std::size_t i = 0; i != initialSearchPoints.size(); ++i){
			if(!function.isFeasible(initialSearchPoints[i]))
				throw SHARKEXCEPTION("[SMS-EMOA::init] starting point(s) not feasible");
			values[i] = function.eval(initialSearchPoints[i]);
		}
		
		std::size_t dim = function.numberOfVariables();
		RealVector lowerBounds(dim, -1E20);
		RealVector upperBounds(dim, 1E20);
		if (function.hasConstraintHandler() && function.getConstraintHandler().isBoxConstrained()) {
			typedef BoxConstraintHandler<SearchPointType> ConstraintHandler;
			ConstraintHandler  const& handler = static_cast<ConstraintHandler const&>(function.getConstraintHandler());
			
			lowerBounds = handler.lower();
			upperBounds = handler.upper();
		} else{
			throw SHARKEXCEPTION("[SMS-EMOA::init] Algorithm does only allow box constraints");
		}
		doInit(initialSearchPoints,values,lowerBounds, upperBounds,mu(),nm(),nc(),crossoverProbability());
	}

	/**
	 * \brief Executes one iteration of the algorithm.
	 * 
	 * \param [in] function The function to iterate upon.
	 */
	void step( ObjectiveFunctionType const& function ) {
		std::vector<IndividualType> offspring = generateOffspring();
		PenalizingEvaluator penalizingEvaluator;
		penalizingEvaluator( function, offspring.begin(), offspring.end() );
		updatePopulation(offspring);
	}
protected:
	/// \brief The individual type of the SMS-EMOA.
	typedef shark::Individual<RealVector,RealVector> IndividualType;

	void doInit(
		std::vector<SearchPointType> const& initialSearchPoints,
		std::vector<ResultType> const& functionValues,
		RealVector const& lowerBounds,
		RealVector const& upperBounds,
		std::size_t mu,
		double nm,
		double nc,
		double crossover_prob
	){
		SIZE_CHECK(initialSearchPoints.size() > 0);
		m_mu = mu;
		m_mutator.m_nm = nm;
		m_crossover.m_nc = nc;
		m_crossoverProbability = crossover_prob;
		m_best.resize( mu );
		m_parents.resize( mu );
		//if the number of supplied points is smaller than mu, fill everything in
		std::size_t numPoints = 0;
		if(initialSearchPoints.size()<=mu){
			numPoints = initialSearchPoints.size();
			for(std::size_t i = 0; i != numPoints; ++i){
				m_parents[i].searchPoint() = initialSearchPoints[i];
				m_parents[i].penalizedFitness() = functionValues[i];
				m_parents[i].unpenalizedFitness() = functionValues[i];
			}
		}
		//copy points randomly
		for(std::size_t i = numPoints; i != mu; ++i){
			std::size_t index = discrete(*mpe_rng,0,initialSearchPoints.size()-1);
			m_parents[i].searchPoint() = initialSearchPoints[index];
			m_parents[i].penalizedFitness() = functionValues[index];
			m_parents[i].unpenalizedFitness() = functionValues[index];
		}
		//create initial mu best points
		for(std::size_t i = 0; i != mu; ++i){
			m_best[i].point = m_parents[i].searchPoint();
			m_best[i].value = m_parents[i].unpenalizedFitness();
		}
		m_selection( m_parents, mu );
		
		m_crossover.init(lowerBounds,upperBounds);
		m_mutator.init(lowerBounds,upperBounds);
	}
	
	std::vector<IndividualType> generateOffspring()const{
		std::vector<IndividualType> offspring(1);
		offspring[0] = createOffspring(m_parents.begin(),m_parents.begin()+mu());
		return offspring;
	}
	
	void updatePopulation(  std::vector<IndividualType> const& offspring) {
		m_parents.push_back(offspring[0]);
		m_selection( m_parents, mu());

		//if the individual got selected, insert it into the parent population
		if(m_parents.back().selected()){
			for(std::size_t i = 0; i != mu(); ++i){
				if(!m_parents[i].selected()){
					m_best[i].point = m_parents[mu()].searchPoint();
					m_best[i].value = m_parents[mu()].unpenalizedFitness();
					m_parents[i] = m_parents.back();
					break;
				}
			}
		}
		m_parents.pop_back();
	}
	
	std::vector<IndividualType> m_parents; ///< Population of size \f$\mu + 1\f$.
private:
	
	IndividualType createOffspring(
		std::vector<IndividualType>::const_iterator begin,
		std::vector<IndividualType>::const_iterator end
	)const{
		TournamentSelection< IndividualType::RankOrdering > selection;

		IndividualType mate1( *selection(*mpe_rng, begin, end ) );
		IndividualType mate2( *selection(*mpe_rng, begin, end) );

		if( coinToss(*mpe_rng, m_crossoverProbability ) ) {
			m_crossover(*mpe_rng, mate1, mate2 );
		}

		if( coinToss(*mpe_rng,0.5) ) {
			m_mutator(*mpe_rng, mate1 );
			return mate1;
		} else {
			m_mutator(*mpe_rng, mate2 );
			return mate2;
		}
	}

	unsigned int m_mu; ///< Size of parent generation

	IndicatorBasedSelection<HypervolumeIndicator> m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.
	SimulatedBinaryCrossover< RealVector > m_crossover; ///< Crossover operator.
	PolynomialMutator m_mutator; ///< Mutation operator.

	double m_crossoverProbability; ///< Crossover probability.
	DefaultRngType* mpe_rng;
};
}


#endif
