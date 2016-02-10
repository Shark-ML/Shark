/*!
 * 
 *
 * \brief       Implements the generational Multi-objective Covariance Matrix Adapation ES.
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_MOCMA
#define SHARK_ALGORITHMS_DIRECT_SEARCH_MOCMA

// MOO specific stuff
#include <shark/Algorithms/DirectSearch/Indicators/HypervolumeIndicator.h>
#include <shark/Algorithms/DirectSearch/Indicators/AdditiveEpsilonIndicator.h>
#include <shark/Algorithms/DirectSearch/Indicators/LeastContributorApproximator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/CMA/CMAIndividual.h>


#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>

#include <boost/foreach.hpp>

namespace shark {

/**
 * \brief Implements the generational MO-CMA-ES.
 *
 * Please see the following papers for further reference:
 *	- Igel, Suttorp and Hansen. Steady-state Selection and Efficient Covariance Matrix Update in the Multi-Objective CMA-ES.
 *	- Voﬂ, Hansen and Igel. Improved Step Size Adaptation for the MO-CMA-ES.
 */
template<typename Indicator=HypervolumeIndicator>
class IndicatorBasedMOCMA : public AbstractMultiObjectiveOptimizer<RealVector >{
public:
	enum NotionOfSuccess{
		IndividualBased,
		PopulationBased
	};
private:
	std::size_t m_mu; ///< Size of parent population
	
	IndicatorBasedSelection<Indicator> m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.
	
	NotionOfSuccess m_notionOfSuccess; ///< Flag for deciding whether the improved step-size adaptation shall be used.
	double m_individualSuccessThreshold;
	double m_initialSigma;
public:
	
	IndicatorBasedMOCMA() {
		m_individualSuccessThreshold = 0.44;
		initialSigma() = 1.0;
		mu() = 100;
		notionOfSuccess() = PopulationBased;
		this->m_features |= AbstractMultiObjectiveOptimizer<RealVector >::CAN_SOLVE_CONSTRAINED;
	}

	/**
	 * \brief Returns the name of the algorithm.
	 */
	std::string name() const {
		return "MOCMA";
	}
	
	std::size_t mu()const{
		return m_mu;
	}
	std::size_t& mu(){
		return m_mu;
	}
	
	double initialSigma()const{
		return m_initialSigma;
	}
	double& initialSigma(){
		return m_initialSigma;
	}
	
	NotionOfSuccess notionOfSuccess()const{
		return m_notionOfSuccess;
	}
	NotionOfSuccess& notionOfSuccess(){
		return m_notionOfSuccess;
	}
	/**
	 * \brief Stores/loads the algorithm's state.
	 * \tparam Archive The type of the archive.
	 * \param [in,out] archive The archive to use for loading/storing.
	 * \param [in] version Currently unused.
	 */
	template<typename Archive>
	void serialize(Archive &archive, const unsigned int version) {
		archive & BOOST_SERIALIZATION_NVP(m_parents);
		archive & BOOST_SERIALIZATION_NVP(m_mu);
		archive & BOOST_SERIALIZATION_NVP(m_best);
		
		archive & BOOST_SERIALIZATION_NVP(m_notionOfSuccess);
		archive & BOOST_SERIALIZATION_NVP(m_individualSuccessThreshold);
		archive & BOOST_SERIALIZATION_NVP(m_initialSigma);
	}
	
	void init( ObjectiveFunctionType& function){
		checkFeatures(function);
		if(!function.canProposeStartingPoint())
			throw SHARKEXCEPTION( "Objective function does not propose a starting point");
		std::vector<RealVector> points(mu());
		for(std::size_t i = 0; i != mu(); ++i){
			points[i] = function.proposeStartingPoint();
		}
		init(function,points);
	}
	/**
	 * \brief Initializes the algorithm for the supplied objective function.
	 * 
	 * \param [in] function The objective function.
	 * \param [in] startingPoints A set of intiial search points.
	 */
	void init( 
		ObjectiveFunctionType& function, 
		std::vector<SearchPointType> const& startingPoints
	){
		checkFeatures(function);
		std::vector<RealVector> values(startingPoints.size());
		for(std::size_t i = 0; i != startingPoints.size(); ++i){
			if(!function.isFeasible(startingPoints[i]))
				throw SHARKEXCEPTION("[MOCMA::init] starting point(s) not feasible");
			values[i] = function.eval(startingPoints[i]);
		}
		this->doInit(startingPoints,values,mu(),initialSigma() );
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
	/// \brief The individual type of the SteadyState-MOCMA.
	typedef CMAIndividual<RealVector> IndividualType;

	void doInit(
		std::vector<SearchPointType> const& startingPoints,
		std::vector<ResultType> const& functionValues,
		std::size_t mu,
		double initialSigma
	){
		m_mu = mu;
		m_initialSigma = initialSigma;
		
		m_best.resize( mu );
		m_parents.resize( mu );
		std::size_t noVariables = startingPoints[0].size();

		//if the number of supplied points is smaller than mu, fill everything in
		std::size_t numPoints = 0;
		if(startingPoints.size()<=mu){
			numPoints = startingPoints.size();
			for(std::size_t i = 0; i != numPoints; ++i){
				m_parents[i] = IndividualType(noVariables,m_individualSuccessThreshold,m_initialSigma);
				m_parents[i].searchPoint() = startingPoints[i];
				m_parents[i].penalizedFitness() = functionValues[i];
				m_parents[i].unpenalizedFitness() = functionValues[i];
			}
		}
		//copy points randomly
		for(std::size_t i = numPoints; i != mu; ++i){
			std::size_t index = Rng::discrete(0,startingPoints.size()-1);
			m_parents[i] = IndividualType(noVariables,m_individualSuccessThreshold,m_initialSigma);
			m_parents[i].searchPoint() = startingPoints[index];
			m_parents[i].penalizedFitness() = functionValues[index];
			m_parents[i].unpenalizedFitness() = functionValues[index];
		}
		//create initial mu best points
		for(std::size_t i = 0; i != mu; ++i){
			m_best[i].point = m_parents[i].searchPoint();
			m_best[i].value = m_parents[i].unpenalizedFitness();
		}
	}
	
	std::vector<IndividualType> generateOffspring()const{
		std::vector<IndividualType> offspring(mu());
		for(std::size_t i = 0; i != mu(); ++i){
			std::size_t parentId = i;
			offspring[i] = m_parents[parentId];
			offspring[i].mutate();
			offspring[i].parent() = parentId;
		}
		return offspring;
	}
	
	void updatePopulation(  std::vector<IndividualType> const& offspringVec) {
		m_parents.insert(m_parents.end(),offspringVec.begin(),offspringVec.end());
		m_selection( m_parents, mu());
		
		//determine from the selection which parent-offspring pair has been successful
		for (std::size_t i = 0; i < mu(); i++) {
			CMAChromosome::IndividualSuccess offspringSuccess = CMAChromosome::Unsuccessful;
			//new update: an offspring is successful if it is selected
			if ( m_notionOfSuccess == PopulationBased && m_parents[mu()+i].selected()) {
				m_parents[mu()+i].updateAsOffspring();
				offspringSuccess = CMAChromosome::Successful;
			}
			//old update: an offspring is successfull if it is better than its parent
			else if ( m_notionOfSuccess == IndividualBased && m_parents[mu()+i].selected() && m_parents[mu()+i].rank() <= m_parents[i].rank()) {
				m_parents[mu()+i].updateAsOffspring();
				offspringSuccess = CMAChromosome::Successful;
			}
			if(m_parents[i].selected()) 
				m_parents[i].updateAsParent(offspringSuccess);
		}
		
		//partition the selected individuals to the front and remove the unselected ones
		std::partition(m_parents.begin(), m_parents.end(),IndividualType::IsSelected);
		m_parents.erase(m_parents.begin()+mu(),m_parents.end());

		//update solution set
		for (std::size_t i = 0; i < mu(); i++) {
			noalias(m_best[i].point) = m_parents[i].searchPoint();
			m_best[i].value = m_parents[i].unpenalizedFitness();
		}
	}
	
	std::vector<IndividualType> m_parents; ///< Population of size \f$\mu + 1\f$.

};

typedef IndicatorBasedMOCMA< HypervolumeIndicator > MOCMA;
typedef IndicatorBasedMOCMA< AdditiveEpsilonIndicator > EpsilonMOCMA;
typedef IndicatorBasedMOCMA< LeastContributorApproximator< FastRng, HypervolumeCalculator > > ApproximatedVolumeMOCMA;

}

#endif
