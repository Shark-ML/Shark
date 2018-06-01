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
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#include <shark/Algorithms/DirectSearch/Operators/Indicators/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Indicators/AdditiveEpsilonIndicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Indicators/CrowdingDistance.h>
#include <shark/Algorithms/DirectSearch/Operators/Indicators/NSGA3Indicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/TournamentSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>


namespace shark {

/// \brief Implements the NSGA-II.
///
/// Please see the following papers for further reference:
///  Deb, Agrawal, Pratap and Meyarivan. 
///  A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II 
///  IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 6, NO. 2, APRIL 2002
/// \ingroup multidirect
template<typename Indicator>
class IndicatorBasedRealCodedNSGAII : public AbstractMultiObjectiveOptimizer<RealVector>{		
public:

	/// \brief Default c'tor.
	IndicatorBasedRealCodedNSGAII(random::rng_type& rng = random::globalRng):mpe_rng(&rng){
		mu() = 100;
		crossoverProbability() = 0.9;
		nc() = 20.0;
		nm() = 20.0;
		this->m_features |= AbstractMultiObjectiveOptimizer<RealVector >::CAN_SOLVE_CONSTRAINED;
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
	
	/// \brief Returns the mutation variation strength parameter.
	double nm()const{
		return m_mutation.m_nm;
	}
	/// \brief Returns a reference to the mutation variation strength parameter.
	double& nm(){
		return m_mutation.m_nm;
	}
	/// \brief Returns the crossover variation strength parameter.
	double nc()const{
		return m_crossover.m_nc;
	}
	/// \brief Returns a reference to the crossover variation strength parameter.
	double& nc(){
		return m_crossover.m_nc;
	}
	
	/// \brief Returns the number of elements in the front.
	std::size_t mu()const{
		return m_mu;
	}
	/// \brief Returns the number of elements in the front.
	std::size_t& mu(){
		return m_mu;
	}
	
	/// \brief The number of points used for initialization of the algorithm.
	std::size_t numInitPoints() const{
		return m_mu;
	}
	
	/// \brief Returns the indicator used.
	Indicator& indicator(){
		return m_selection.indicator();
	}
	/// \brief Returns the indicator used.
	Indicator const& indicator()const{
		return m_selection.indicator();
	}

	void read( InArchive & archive ){
		archive >> BOOST_SERIALIZATION_NVP(m_parents);
		archive >> BOOST_SERIALIZATION_NVP(m_mu);
		archive >> BOOST_SERIALIZATION_NVP(m_best);
		
		archive >> BOOST_SERIALIZATION_NVP(m_selection);
		archive >> BOOST_SERIALIZATION_NVP(m_crossover);
		archive >> BOOST_SERIALIZATION_NVP(m_mutation);
		archive >> BOOST_SERIALIZATION_NVP(m_crossoverProbability);
	}
	void write( OutArchive & archive ) const{
		archive << BOOST_SERIALIZATION_NVP(m_parents);
		archive << BOOST_SERIALIZATION_NVP(m_mu);
		archive << BOOST_SERIALIZATION_NVP(m_best);
		
		archive << BOOST_SERIALIZATION_NVP(m_selection);
		archive << BOOST_SERIALIZATION_NVP(m_crossover);
		archive << BOOST_SERIALIZATION_NVP(m_mutation);
		archive << BOOST_SERIALIZATION_NVP(m_crossoverProbability);
	}
	
	using AbstractMultiObjectiveOptimizer<RealVector >::init;
	
	/// \brief Initializes the algorithm for the supplied objective function.
	///
	/// \param [in] function The objective function.
	/// \param [in] initialSearchPoints A set of intiial search points.
	void init( 
		ObjectiveFunctionType const& function, 
		std::vector<SearchPointType> const& initialSearchPoints
	){
		checkFeatures(function);
		std::vector<RealVector> values(initialSearchPoints.size());
		for(std::size_t i = 0; i != initialSearchPoints.size(); ++i){
			SHARK_RUNTIME_CHECK(function.isFeasible(initialSearchPoints[i]),"Starting point(s) not feasible");
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
			SHARK_RUNTIME_CHECK(
				!function.isConstrained(), "Algorithm does only allow box constraints"
			);
		}
		
		doInit(initialSearchPoints,values,lowerBounds, upperBounds, mu(), nm(), nc(), crossoverProbability());
	}
	
	/// \brief Executes one iteration of the algorithm.
	///
	///\param [in] function The function to iterate upon.
	void step( ObjectiveFunctionType const& function ) {
		std::vector<IndividualType> offspring = generateOffspring();
		PenalizingEvaluator penalizingEvaluator;
		penalizingEvaluator( function, offspring.begin(), offspring.end() );
		updatePopulation(offspring);
	}
protected:
	/// \brief The individual type of the NSGA-II.
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
		m_mutation.m_nm = nm;
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
			std::size_t index = random::discrete(*mpe_rng, std::size_t(0),initialSearchPoints.size()-1);
			m_parents[i].searchPoint() = initialSearchPoints[index];
			m_parents[i].penalizedFitness() = functionValues[index];
			m_parents[i].unpenalizedFitness() = functionValues[index];
		}
		//create initial mu best points
		for(std::size_t i = 0; i != mu; ++i){
			m_best[i].point = m_parents[i].searchPoint();
			m_best[i].value = m_parents[i].unpenalizedFitness();
		}
		m_crossover.init(lowerBounds,upperBounds);
		m_mutation.init(lowerBounds,upperBounds);
		indicator().init(functionValues.front().size(),mu,*mpe_rng);
		m_selection( m_parents, mu );
		
		
	}
	
	std::vector<IndividualType> generateOffspring()const{
		TournamentSelection< IndividualType::RankOrdering > selection;
		std::vector<IndividualType> offspring(mu());
		selection(
			*mpe_rng,
			m_parents.begin(), 
			m_parents.end(),
			offspring.begin(),
			offspring.end()
		);

		for( std::size_t i = 0; i < mu()-1; i+=2 ) {
			if( random::coinToss(*mpe_rng, m_crossoverProbability ) ) {
				m_crossover(*mpe_rng,offspring[i], offspring[i +1] );
			}
		}
		for( std::size_t i = 0; i < mu(); i++ ) {
			m_mutation(*mpe_rng, offspring[i] );
		}
		return offspring;
	}

	void updatePopulation(  std::vector<IndividualType> const& offspringVec) {
		m_parents.insert(m_parents.end(),offspringVec.begin(),offspringVec.end());
		m_selection( m_parents, mu());
		
		//partition the selected individuals to the front and remove the unselected ones
		std::partition(m_parents.begin(), m_parents.end(), [](IndividualType const& ind){return ind.selected();});
		m_parents.erase(m_parents.begin()+mu(),m_parents.end());

		//update solution set
		for (std::size_t i = 0; i < mu(); i++) {
			noalias(m_best[i].point) = m_parents[i].searchPoint();
			m_best[i].value = m_parents[i].unpenalizedFitness();
		}
	}

	std::vector<IndividualType> m_parents; ///< Population of size \f$\mu + 1\f$.
	random::rng_type* mpe_rng; 
private:
	std::size_t m_mu; ///< Size of parent generation

	IndicatorBasedSelection<Indicator> m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.

	SimulatedBinaryCrossover< RealVector > m_crossover; ///< Crossover operator.
	PolynomialMutator m_mutation; ///< Mutation operator.

	double m_crossoverProbability; ///< Crossover probability.
};

typedef IndicatorBasedRealCodedNSGAII< HypervolumeIndicator > RealCodedNSGAII;
typedef IndicatorBasedRealCodedNSGAII< AdditiveEpsilonIndicator > EpsRealCodedNSGAII;
typedef IndicatorBasedRealCodedNSGAII< CrowdingDistance > CrowdingRealCodedNSGAII;
}
#endif
