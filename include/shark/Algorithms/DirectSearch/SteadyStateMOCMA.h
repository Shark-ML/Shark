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
#include <shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/CMA/CMAIndividual.h>

#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>

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
public:
	
	IndicatorBasedSteadyStateMOCMA(DefaultRngType& rng = Rng::globalRng):mpe_rng(&rng){
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
		return "SteadyStateMOCMA";
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
	
	Indicator& indicator(){
		return m_selection.indicator();
	}
	Indicator const& indicator()const{
		return m_selection.indicator();
	}
	
	void read( InArchive & archive ){
		archive >> BOOST_SERIALIZATION_NVP(m_parents);
		archive >> BOOST_SERIALIZATION_NVP(m_mu);
		archive >> BOOST_SERIALIZATION_NVP(m_best);
		
		archive >> BOOST_SERIALIZATION_NVP(m_selection);
		archive >> BOOST_SERIALIZATION_NVP(m_notionOfSuccess);
		archive >> BOOST_SERIALIZATION_NVP(m_individualSuccessThreshold);
		archive >> BOOST_SERIALIZATION_NVP(m_initialSigma);
	}
	void write( OutArchive & archive ) const{
		archive << BOOST_SERIALIZATION_NVP(m_parents);
		archive << BOOST_SERIALIZATION_NVP(m_mu);
		archive << BOOST_SERIALIZATION_NVP(m_best);
		
		archive << BOOST_SERIALIZATION_NVP(m_selection);
		archive << BOOST_SERIALIZATION_NVP(m_notionOfSuccess);
		archive << BOOST_SERIALIZATION_NVP(m_individualSuccessThreshold);
		archive << BOOST_SERIALIZATION_NVP(m_initialSigma);
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
				throw SHARKEXCEPTION("[SteadyStateMOCMA::init] starting point(s) not feasible");
			values[i] = function.eval(initialSearchPoints[i]);
		}
		this->doInit(initialSearchPoints,values,mu(),initialSigma() );
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
		std::vector<SearchPointType> const& initialSearchPoints,
		std::vector<ResultType> const& functionValues,
		std::size_t mu,
		double initialSigma
	){
		SIZE_CHECK(initialSearchPoints.size() > 0);
		
		m_mu = mu;
		m_initialSigma = initialSigma;
		
		m_best.resize( mu );
		m_parents.resize( mu );
		std::size_t noVariables = initialSearchPoints[0].size();

		//if the number of supplied points is smaller than mu, fill everything in
		std::size_t numPoints = 0;
		if(initialSearchPoints.size()<=mu){
			numPoints = initialSearchPoints.size();
			for(std::size_t i = 0; i != numPoints; ++i){
				m_parents[i] = IndividualType(noVariables,m_individualSuccessThreshold,m_initialSigma);
				m_parents[i].searchPoint() = initialSearchPoints[i];
				m_parents[i].penalizedFitness() = functionValues[i];
				m_parents[i].unpenalizedFitness() = functionValues[i];
			}
		}
		//copy points randomly
		for(std::size_t i = numPoints; i != mu; ++i){
			std::size_t index = discrete(*mpe_rng, 0,initialSearchPoints.size()-1);
			m_parents[i] = IndividualType(noVariables,m_individualSuccessThreshold,m_initialSigma);
			m_parents[i].searchPoint() = initialSearchPoints[index];
			m_parents[i].penalizedFitness() = functionValues[index];
			m_parents[i].unpenalizedFitness() = functionValues[index];
		}
		//create initial mu best points
		for(std::size_t i = 0; i != mu; ++i){
			m_best[i].point = m_parents[i].searchPoint();
			m_best[i].value = m_parents[i].unpenalizedFitness();
		}
		m_selection(m_parents,mu);
		sortRankOneToFront();
	}
	
	std::vector<IndividualType> generateOffspring()const{
		//find the last element with rank 1
		std::size_t maxIdx = 0;
		for (; maxIdx < m_parents.size(); maxIdx++) {
			if (m_parents[maxIdx].rank() != 1)
				break;
		}
		//sample a random parent with rank 1
		std::size_t parentId = discrete(*mpe_rng, 0, maxIdx-1);
		std::vector<IndividualType> offspring;
		offspring.push_back(m_parents[parentId]);
		offspring[0].mutate(*mpe_rng);
		offspring[0].parent() = parentId;
		return offspring;
	}
	
	void updatePopulation(  std::vector<IndividualType> const& offspringVec) {
		m_parents.push_back(offspringVec[0]);
		m_selection( m_parents, mu());
		
		IndividualType& offspring = m_parents.back();
		IndividualType& parent = m_parents[offspring.parent()];
		
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

		//if the individual got selected, insert it into the parent population
		if(m_parents.back().selected()){
			for(std::size_t i = 0; i != mu(); ++i){
				if(!m_parents[i].selected()){
					m_best[i].point = m_parents.back().searchPoint();
					m_best[i].value = m_parents.back().unpenalizedFitness();
					m_parents[i] = m_parents.back();
					break;
				}
			}
		}
		m_parents.pop_back();
		sortRankOneToFront();
	}
	
	std::vector<IndividualType> m_parents; ///< Population of size \f$\mu + 1\f$.
private:
	std::size_t m_mu; ///< Size of parent population
	
	IndicatorBasedSelection<Indicator> m_selection; ///< Selection operator relying on the (contributing) hypervolume indicator.
	
	NotionOfSuccess m_notionOfSuccess; ///< Flag for deciding whether the improved step-size adaptation shall be used.
	double m_individualSuccessThreshold;
	double m_initialSigma;

	/// \brief sorts all individuals with rank one to the front
	void sortRankOneToFront(){
		std::size_t start = 0;
		std::size_t end = mu()-1;
		while(start != end){
			if(m_parents[start].rank() == 1){
				++start;
			}else if(m_parents[end].rank() != 1){
				--end;
			}else{
				swap(m_parents[start],m_parents[end]);
				swap(m_best[start],m_best[end]);
			}
		}
	}
	
	DefaultRngType* mpe_rng;
};

typedef IndicatorBasedSteadyStateMOCMA< HypervolumeIndicator > SteadyStateMOCMA;
typedef IndicatorBasedSteadyStateMOCMA< AdditiveEpsilonIndicator > EpsilonSteadyStateMOCMA;

}

#endif
