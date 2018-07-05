/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_UNSUPERVISED_RBM_SAMPLING_TEMPEREDMARKOVCHAIN_H
#define SHARK_UNSUPERVISED_RBM_SAMPLING_TEMPEREDMARKOVCHAIN_H

#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>
#include <shark/Core/Random.h>
#include <shark/Unsupervised/RBM/Tags.h>
#include <vector>
#include "Impl/SampleTypes.h"
namespace shark{
	

//\brief models a set of tempered Markov chains given a TransitionOperator.
// e.g.  TemperedMarkovChain<GibbsOperator<RBM> > chain, leads to the set of chains
// used for parallel tempering. 
template<class Operator>
class TemperedMarkovChain{
private:
	typedef typename Operator::HiddenSample HiddenSample;
	typedef typename Operator::VisibleSample VisibleSample;
public:

	///\brief The MarkovChain can't be used to compute several samples at once.
	///
	/// The tempered markov chain ues it's batch capabilities allready to compute the samples for all temperatures
	/// At the same time. Also it is much more powerfull when all samples are drawn one after another for a higher mixing rate.
	static const bool computesBatch = false;

	///\brief The type of the RBM the operator is working with.
	typedef typename Operator::RBM RBM;
	
	///\brief A batch of samples containing hidden and visible samples as well as the energies.
	typedef detail::MarkovChainSample<HiddenSample,VisibleSample> Sample;
	
private:
	Sample m_temperedChains;
	RealVector m_betas;
	Operator m_operator;
	
	void metropolisSwap(std::size_t i, std::size_t j){
		RealVector const& baseRate = transitionOperator().rbm()->visibleNeurons().baseRate();
		double betaDiff = beta(i) - beta(j);
		double energyDiff = m_temperedChains.energy(i) - m_temperedChains.energy(j); 
		double baseRateDiff = inner_prod(row(m_temperedChains.visible.state,i) - row(m_temperedChains.visible.state,j),baseRate);
		double r = betaDiff * energyDiff + betaDiff*baseRateDiff;
		
		double z = random::uni(m_operator.rbm()->rng(),0,1);
		if( r >= 0 || (z > 0 && std::log(z) < r) ){
			m_temperedChains.swap_rows(i,j);
		}
	}

public:
	TemperedMarkovChain(RBM* rbm):m_operator(rbm){}
	
	const Operator& transitionOperator()const{
		return m_operator;
	}
	Operator& transitionOperator(){
		return m_operator;
	}
	

	/// \brief Sets the number of temperatures and initializes the tempered chains accordingly. 
	///
	/// @param temperatures number of temperatures  
	void setNumberOfTemperatures(std::size_t temperatures){
		std::size_t visibles=m_operator.rbm()->numberOfVN();
		std::size_t hiddens=m_operator.rbm()->numberOfHN();
		m_temperedChains = Sample(temperatures,hiddens,visibles);
		m_betas.resize(temperatures);
	}
	
	/// \brief Sets the number of temperatures and initializes them in a uniform spacing
	///
	/// Temperatures are spaced equally between 0 and 1.
	/// @param temperatures number of temperatures  
	void setUniformTemperatureSpacing(std::size_t temperatures){
		setNumberOfTemperatures(temperatures);
		for(std::size_t i = 0; i != temperatures; ++i){
			double factor = temperatures - 1.0;
			setBeta(i,1.0 - i/factor);
		}	
	}


	/// \brief Returns the number Of temperatures.
	std::size_t numberOfTemperatures()const{
		return m_betas.size();
	}
	
	void setBatchSize(std::size_t batchSize){
		SHARK_RUNTIME_CHECK(batchSize == 1, "Markov chain can only compute batches of size 1.");
	}
	std::size_t batchSize(){
		return 1;
	}
	
	void setBeta(std::size_t i, double beta){
		SIZE_CHECK(i < m_betas.size());
		m_betas(i) = beta;
	}
	
	double beta(std::size_t i)const{
		SIZE_CHECK(i < m_betas.size());
		return m_betas(i);
	}
	
	RealVector const& beta()const{
		return m_betas;
	}
	
	///\brief Returns the current state of the chain for all beta values.
	Sample const& samples()const{
		return m_temperedChains;
	}
	
	/// \brief Returns the current batch of samples of the Markov chain. 
	Sample& samples(){
		return m_temperedChains;
	}

	///\brief Initializes the markov chain using samples drawn uniformly from the set.
	///
	/// Be aware that the number of chains and the temperatures need to bee specified previously.
	/// @param dataSet the data set
	void initializeChain(Data<RealVector> const& dataSet){
		SHARK_RUNTIME_CHECK(m_temperedChains.size() != 0,"You did not initialize the number of temperatures bevor initializing the chain!");

		RealMatrix sampleData = randomSubBatch(elements(dataSet),m_temperedChains.size());
		initializeChain(sampleData);
	}
	
	/// \brief Initializes with data points from a batch of points
	///
	/// Be aware that the number of chains and the temperatures need to bee specified previously.
	/// @param sampleData the data set
	void initializeChain(RealMatrix const& sampleData){
 		SHARK_RUNTIME_CHECK(m_temperedChains.size() != 0,"You did not initialize the number of temperatures bevor initializing the chain!");

		m_operator.createSample(m_temperedChains.hidden,m_temperedChains.visible,sampleData,m_betas);
		
		m_temperedChains.energy = m_operator.calculateEnergy(
			m_temperedChains.hidden, m_temperedChains.visible 
		);
	}
	//updates the chain using the current sample
	void step(unsigned int k){
		for(std::size_t i = 0; i != k; ++i){
			//do one step of the tempered the Markov chains at the same time
			m_operator.stepVH(m_temperedChains.hidden, m_temperedChains.visible,1,m_betas);
			
			//calculate energy for samples at all temperatures
			m_temperedChains.energy = m_operator.calculateEnergy(
				m_temperedChains.hidden,
				m_temperedChains.visible 
			);

			//EVEN phase
			std::size_t elems = m_temperedChains.size();
			for(std::size_t i = 0; i < elems-1; i+=2){
				metropolisSwap(i, i+1);
			}
			//ODD phase
			for(std::size_t i = 1; i < elems-1; i+=2){
				metropolisSwap(i, i+1);
			}
			m_operator.rbm()->hiddenNeurons().sufficientStatistics(
				m_temperedChains.hidden.input,m_temperedChains.hidden.statistics, m_betas
			);
		}
	}
};
	
}
#endif
