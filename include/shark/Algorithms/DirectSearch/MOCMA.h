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
template<typename Indicator>
class IndicatorBasedMOCMA : public AbstractMultiObjectiveOptimizer<RealVector >{
public:
	
	enum NotionOfSuccess{
		IndividualBased,
		PopulationBased
	};
	
	std::vector<CMAIndividual<RealVector> > m_pop; ///< Population of size \f$\mu+\mu\f$.
	unsigned int m_mu;///< Size of parent generation
	
	shark::PenalizingEvaluator m_evaluator; ///< Evaluation operator.
	IndicatorBasedSelection< Indicator > m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.

	NotionOfSuccess m_notionOfSuccess; ///< Flag for deciding whether the improved step-size adaptation shall be used.
	double m_individualSuccessThreshold;
	double m_initialSigma;

	/**
	 * \brief Default c'tor.
	 */
	IndicatorBasedMOCMA() {
		m_individualSuccessThreshold = 0.44;
		initialSigma() = 1.0;
		constrainedPenaltyFactor() = 1E-6;
		mu() = 100;
		notionOfSuccess() = PopulationBased;
	}

	/**
	 * \brief Returns the name of the algorithm.
	 */
	std::string name() const {
		return "MOCMA";
	}
	
	unsigned int mu()const{
		return m_mu;
	}
	unsigned int& mu(){
		return m_mu;
	}
	
	double initialSigma()const{
		return m_initialSigma;
	}
	double& initialSigma(){
		return m_initialSigma;
	}
	
	/// \brief Returns the penalty factor for an individual that is outside the feasible area.
	///
	/// The value is multiplied with the distance to the nearest feasible point.
	double constrainedPenaltyFactor()const{
		return m_evaluator.m_penaltyFactor;
	}
	
	/// \brief Returns a reference to the penalty factor for an individual that is outside the feasible area.
	///
	/// The value is multiplied with the distance to the nearest feasible point.
	double& constrainedPenaltyFactor(){
		return m_evaluator.m_penaltyFactor;
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
		archive & BOOST_SERIALIZATION_NVP(m_pop);
		archive & BOOST_SERIALIZATION_NVP(m_mu);
		archive & BOOST_SERIALIZATION_NVP(m_best);
		archive & BOOST_SERIALIZATION_NVP(m_evaluator);
		archive & BOOST_SERIALIZATION_NVP(m_notionOfSuccess);
		archive & BOOST_SERIALIZATION_NVP(m_individualSuccessThreshold);
		archive & BOOST_SERIALIZATION_NVP(m_initialSigma);
	}

	/**
	 * \brief Initializes the algorithm from a configuration-tree node.
	 *
	 * The following sub keys are recognized:
	 *	- Mu, type: unsigned int, default value: 100.
	 *	- PenaltyFactor, type: double, default value: \f$10^{-6}\f$.
	 *	- NotionOfSuccess, type: string, default value: PopulationBased. Can also be IndividualBased.
	 *	- initialSigma, an initial estimate for the diagonal of the covariance matrix.
	 *
	 * \param [in] node The configuration tree node.
	 */
	void configure(PropertyTree const& node) {
		mu() = node.get("Mu", mu());
		constrainedPenaltyFactor() = node.get("PenaltyFactor", constrainedPenaltyFactor());
		initialSigma() = node.get("InitialSigma",initialSigma());
		if(node.find("NotionOfSuccess") != node.not_found()){
			std::string str = node.get<std::string>("NotionOfSuccess");
			if(str == "IndividualBased"){
				notionOfSuccess() = IndividualBased;
			}else if(str == "PopulationBased"){
				notionOfSuccess() = PopulationBased;
			}else{
				throw SHARKEXCEPTION("Unknown Value for NotionOfSuccess:"+str);
			}
		}
	}

	using AbstractMultiObjectiveOptimizer<RealVector >::init;
	/**
	 * \brief Initializes the algorithm for the supplied objective function.
	 * \tparam ObjectiveFunction The type of the objective function,
	 * needs to adhere to the concept of an AbstractObjectiveFunction.
	 * \param [in] f The objective function.
	 * \param [in] startingPoints A set of intiial search points.
	 */
	void init( 
		ObjectiveFunctionType const& function, 
		std::vector<SearchPointType> const& startingPoints
	){
		m_pop.resize(2 * mu());
		m_best.resize(mu());
		std::size_t noVariables = function.numberOfVariables();
		
		for(std::size_t i = 0; i != mu(); ++i){
			CMAIndividual<RealVector> individual(noVariables,m_individualSuccessThreshold,m_initialSigma);
			function.proposeStartingPoint(individual.searchPoint());
			m_pop[i] = individual;
		}
		//valuate and create first front
		m_evaluator(function, m_pop.begin(),m_pop.begin()+mu());
		for(std::size_t i = 0; i != mu(); ++i){
			m_best[i].point = m_pop[i].searchPoint();
			m_best[i].value = m_pop[i].unpenalizedFitness();
		}
	}

	/**
	 * \brief Executes one iteration of the algorithm.
	 * 
	 * \param [in] function The function to iterate upon.
	 */
	void step( ObjectiveFunctionType const& function ) {

		//generate new offspring, evaluate and select
		for (std::size_t i = 0; i < mu(); i++) {
			m_pop[mu()+i] = m_pop[i];
			m_pop[mu()+i].mutate();
			m_evaluator(function, m_pop[mu()+i]);
		}
		m_evaluator(function, m_pop.begin()+mu(),m_pop.end());
		m_selection(m_pop,m_mu);
		
		//determine from the selection which parent-offspring pair has been successfull
		for (std::size_t i = 0; i < mu(); i++) {
			CMAChromosome::IndividualSuccess offspringSuccess = CMAChromosome::Unsuccessful;
			//new update: an offspring is successfull if it is selected
			if ( m_notionOfSuccess == PopulationBased && m_pop[mu()+i].selected()) {
				m_pop[mu()+i].updateAsOffspring();
				offspringSuccess = CMAChromosome::Successful;
			}
			//old update: an offspring is successfull if it is better than its parent
			else if ( m_notionOfSuccess == IndividualBased && m_pop[mu()+i].selected() && m_pop[mu()+i].rank() <= m_pop[i].rank()) {
				m_pop[mu()+i].updateAsOffspring();
				offspringSuccess = CMAChromosome::Successful;
			}
			if(m_pop[i].selected()) 
				m_pop[i].updateAsParent(offspringSuccess);
		}
		
		//partition the selected individuals to the front
		std::partition(m_pop.begin(), m_pop.end(),CMAIndividual<RealVector>::IsSelected);

		//update individuals and generate solution set
		for (unsigned int i = 0; i < mu(); i++) {
			noalias(m_best[i].point) = m_pop[i].searchPoint();
			m_best[i].value = m_pop[i].unpenalizedFitness();
		}
	};

};

typedef IndicatorBasedMOCMA< HypervolumeIndicator > MOCMA;
typedef IndicatorBasedMOCMA< AdditiveEpsilonIndicator > EpsilonMOCMA;
typedef IndicatorBasedMOCMA< LeastContributorApproximator< FastRng, HypervolumeCalculator > > ApproximatedVolumeMOCMA;

}

#endif
