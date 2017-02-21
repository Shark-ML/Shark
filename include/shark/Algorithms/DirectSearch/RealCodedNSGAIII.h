/*!
 * 
 *
 * \brief       RealCodedNSGAIIII.h
 * 
 * 
 *
 * \author      O.Krause
 * \date        2016
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_REAL_CODED_NSGA_III_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_REAL_CODED_NSGA_III_H

#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>

#include <shark/Algorithms/DirectSearch/Individual.h>

// MOO specific stuff
#include <shark/Algorithms/DirectSearch/Operators/Indicators/NSGA3Indicator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/TournamentSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>


namespace shark {

/// \brief Implements the NSGA-III
///
/// Please see the following papers for further reference:
/// Deb, Kalyanmoy, and Himanshu Jain. 
/// "An evolutionary many-objective optimization algorithm using 
/// reference-point-based nondominated sorting approach, 
/// part I: Solving problems with box constraints."
/// IEEE Trans. Evolutionary Computation 18.4 (2014): 577-601.
class RealCodedNSGAIII : public AbstractMultiObjectiveOptimizer<RealVector>{		
public:

	/// \brief Default c'tor.
	RealCodedNSGAIII(DefaultRngType& rng = Rng::globalRng):mpe_rng(&rng){
		mu() = 100;
		crossoverProbability() = 0.9;
		nc() = 20.0;
		nm() = 20.0;
		this->m_features |= AbstractMultiObjectiveOptimizer<RealVector >::CAN_SOLVE_CONSTRAINED;
	}

	std::string name() const {
		return "RealCodedNSGAIII";
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
	/// \brief Returns the indicator used.
	NSGA3Indicator& indicator(){
		return m_selection.indicator();
	}
	/// \brief Returns the indicator used.
	NSGA3Indicator const& indicator()const{
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

	/// \brief Initializes the algorithm for the supplied objective function.
	///
	/// \param [in] function The objective function.
	void init( ObjectiveFunctionType& function){
		checkFeatures(function);
		if(!function.canProposeStartingPoint())
			throw SHARKEXCEPTION( "[RealCodedNSGAIII::init] Objective function does not propose a starting point");
		std::vector<RealVector> points(mu());
		for(std::size_t i = 0; i != mu(); ++i){
			points[i] = function.proposeStartingPoint();
		}
		init(function,points);
	}
	/// \brief Initializes the algorithm for the supplied objective function.
	///
	/// \param [in] function The objective function.
	/// \param [in] initialSearchPoints A set of intiial search points.
	void init( 
		ObjectiveFunctionType& function, 
		std::vector<SearchPointType> const& initialSearchPoints
	){
		checkFeatures(function);
		std::vector<RealVector> values(initialSearchPoints.size());
		for(std::size_t i = 0; i != initialSearchPoints.size(); ++i){
			if(!function.isFeasible(initialSearchPoints[i]))
				throw SHARKEXCEPTION("[RealCodedNSGAIII::init] starting point(s) not feasible");
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
			throw SHARKEXCEPTION("[RealCodedNSGAII::init] Algorithm does only allow box constraints");
		}
		
		std::vector<RealVector> Z;
		for(std::size_t i = 0; i != mu(); ++i){
			RealVector z(function.numberOfObjectives());
			if(z.size() == 2){
				z(0) = double(i)/(mu()-1.0);
				z(1) = 1 - z(0);
			}else if(i < function.numberOfObjectives()){
				z.clear();
				z(i) = 1;
			}else {
				for(auto& val:z){
					val = uni(*mpe_rng,0,1);
				}
			}
			Z.push_back(z);
		}
		
		doInit(initialSearchPoints,values,lowerBounds, upperBounds, mu(), nm(), nc(), crossoverProbability(), Z);
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
	/// \brief The individual type of the NSGA-III.
	typedef shark::Individual<RealVector,RealVector> IndividualType;

	void doInit(
		std::vector<SearchPointType> const& initialSearchPoints,
		std::vector<ResultType> const& functionValues,
		RealVector const& lowerBounds,
		RealVector const& upperBounds,
		std::size_t mu,
		double nm,
		double nc,
		double crossover_prob,
		std::vector<RealVector> const& Z
	){
		SIZE_CHECK(initialSearchPoints.size() > 0);
		SIZE_CHECK(lowerBounds.size() == upperBounds.size());
		SIZE_CHECK(Z.size() == mu);
		
		m_mu = mu;
		m_mutation.m_nm = nm;
		m_crossover.m_nc = nc;
		m_crossoverProbability = crossover_prob;
		m_best.resize( mu );
		m_parents.resize( mu );
		indicator().setReferencePoints(Z);
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
			std::size_t index = discrete(*mpe_rng, 0,initialSearchPoints.size()-1);
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
		m_mutation.init(lowerBounds,upperBounds);
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
			if( coinToss(*mpe_rng, m_crossoverProbability ) ) {
				m_crossover(*mpe_rng,offspring[i], offspring[i +1] );
			}
		}
		for( std::size_t i = 0; i < mu(); i++ ) {
			m_mutation(*mpe_rng, offspring[i] );
		}
		return offspring;
	}

	void updatePopulation( std::vector<IndividualType> const& offspringVec) {
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
private:
	std::vector<IndividualType> m_parents; ///< Population of size \f$\mu + 1\f$.
	std::size_t m_mu; ///< Size of parent generation
	RealMatrix m_referencePoints; ///< Trade-Off points used by the algorithm

	IndicatorBasedSelection<NSGA3Indicator> m_selection;

	SimulatedBinaryCrossover< RealVector > m_crossover; ///< Crossover operator.
	PolynomialMutator m_mutation; ///< Mutation operator.

	double m_crossoverProbability; ///< Crossover probability.
	DefaultRngType* mpe_rng; 
};

}
#endif
