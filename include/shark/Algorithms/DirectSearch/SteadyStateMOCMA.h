/*!
 * 
 *
 * \brief       SteadyStateMOCMA.h
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_STEADYSTATEMOCMA
#define SHARK_ALGORITHMS_DIRECT_SEARCH_STEADYSTATEMOCMA

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
 * \brief Implements the \f$(\mu+1)\f$-MO-CMA-ES.
 *
 * Please see the following papers for further reference:
 *	- Igel, Suttorp and Hansen. Steady-state Selection and Efficient Covariance Matrix Update in the Multi-Objective CMA-ES.
 *	- Voﬂ, Hansen and Igel. Improved Step Size Adaptation for the MO-CMA-ES.
 */
template<typename Indicator=HypervolumeIndicator>
class IndicatorBasedSteadyStateMOCMA : public AbstractMultiObjectiveOptimizer<RealVector >{
public:
	enum NotionOfSuccess{
		IndividualBased,
		PopulationBased
	};
private:

	typedef CMAIndividual<RealVector> Individual;
	std::vector< Individual > m_pop; ///< Population of size \f$\mu+1\f$.
	unsigned int m_mu; ///< Size of parent population
	
	shark::PenalizingEvaluator m_evaluator; ///< Evaluation operator.
	IndicatorBasedSelection<Indicator> m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.
	
	NotionOfSuccess m_notionOfSuccess; ///< Flag for deciding whether the improved step-size adaptation shall be used.
	double m_individualSuccessThreshold;
	double m_initialSigma;
public:
	
	IndicatorBasedSteadyStateMOCMA() {
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
		return "SteadyStateMOCMA";
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
		m_pop.resize(mu());
		m_best.resize(mu());
		std::size_t noVariables = function.numberOfVariables();
		
		for(std::size_t i = 0; i != mu(); ++i){
			m_pop[i] = Individual(noVariables,m_individualSuccessThreshold,m_initialSigma);
			function.proposeStartingPoint(m_pop[i].searchPoint());
			m_evaluator(function, m_pop[i]);
			m_best[i].point = m_pop[i].searchPoint();
			m_best[i].value = m_pop[i].unpenalizedFitness();
		}
		m_selection(m_pop,m_mu);
		m_pop.push_back(Individual(noVariables,m_individualSuccessThreshold,m_initialSigma));
		sortRankOneToFront();
	}

	/**
	 * \brief Executes one iteration of the algorithm.
	 * 
	 * \param [in] function The function to iterate upon.
	 */
	void step( ObjectiveFunctionType const& function ) {
		//find the last element with rank 1
		int maxIdx = 0;
		for (unsigned int i = 0; i < m_pop.size(); i++) {
			if (m_pop[i].rank() != 1)
				break;
			maxIdx = i;
		}
		//sample a random parent with rank 1
		Individual& parent = m_pop[Rng::discrete(0, std::max(0, maxIdx-1))];
		//sample offspring from this parent
		Individual& offspring = m_pop[mu()];
		offspring = parent;
		offspring.mutate();
		m_evaluator(function, offspring);

		m_selection(m_pop,m_mu);
		if (m_notionOfSuccess == IndividualBased && offspring.selected()) {
			offspring.updateAsOffspring();
			parent.updateAsParent(CMAChromosome::Successful);
		}
		else if (m_notionOfSuccess == PopulationBased && offspring.selected() && offspring.rank() <= parent.rank() ) {
			offspring.updateAsOffspring();
			parent.updateAsParent(CMAChromosome::Successful);
		}else{
			parent.updateAsParent(CMAChromosome::Unsuccessful);
		}

		if(offspring.selected()){
			//exchange the selected and nonselected elements
			for(std::size_t i = 0; i != mu(); ++i){
				if(!m_pop[i].selected()){
					noalias(m_best[i].point) = m_pop[mu()].searchPoint();
					m_best[i].value = m_pop[mu()].unpenalizedFitness();
					swap(m_pop[i] , m_pop[mu()]);
				}
			}
			//now push rank 1 elements to the front
			sortRankOneToFront();
		}
	}
protected:
	/// \brief sorts all individuals with rank one to the front
	void sortRankOneToFront(){
		std::size_t start = 0;
		std::size_t end = mu()-1;
		while(start != end){
			if(m_pop[start].rank() == 1){
				++start;
			}else if(m_pop[end].rank() != 1){
				--end;
			}else{
				swap(m_pop[start],m_pop[end]);
				swap(m_best[start],m_best[end]);
			}
		}
	}
};

typedef IndicatorBasedSteadyStateMOCMA< HypervolumeIndicator > SteadyStateMOCMA;
typedef IndicatorBasedSteadyStateMOCMA< LeastContributorApproximator< FastRng, HypervolumeCalculator > > ApproximatedVolumeSteadyStateMOCMA;
typedef IndicatorBasedSteadyStateMOCMA< AdditiveEpsilonIndicator > EpsilonSteadyStateMOCMA;

}

#endif
