/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_UNSUPERVISED_RBM_SAMPLING_TEMPEREDMARKOVCHAIN_H
#define SHARK_UNSUPERVISED_RBM_SAMPLING_TEMPEREDMARKOVCHAIN_H

#include <shark/Data/Dataset.h>
#include <shark/Rng/DiscreteUniform.h>
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
	typedef typename Batch<detail::MarkovChainSample<HiddenSample,VisibleSample> >::type SampleBatch;
	
	///\brief Mutable reference to an element of the batch.
	typedef typename SampleBatch::reference reference;
	
	///\brief Immutable reference to an element of the batch.
	typedef typename SampleBatch::const_reference const_reference;
	
private:
	SampleBatch m_temperedChains;
	RealVector m_betas;
	Operator m_operator;
	
	void metropolisSwap(reference low, double betaLow, reference high, double betaHigh){
		double betaDiff = betaLow - betaHigh;
		double energyDiff = low.energy - high.energy; 
		double r = betaDiff * energyDiff;
		Uniform<typename RBM::RngType> uni(m_operator.rbm()->rng(),0,1);
		double z = uni();
		if( r >= 0 || (z > 0 && std::log(z) < r) ){
			swap(high,low);
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
		m_temperedChains = SampleBatch(temperatures,visibles,hiddens);
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
		SHARK_CHECK(batchSize == 1, "[TemperedMarkovChain::setBatchSize] markov chain can only compute batches of size 1.");
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
	
	///\brief Returns the current state of the chain for beta = 1.
	const_reference sample()const{
		return const_reference(m_temperedChains,0);
	}
	///\brief Returns the current state of the chain for all beta values.
	SampleBatch const& samples()const{
		return m_temperedChains;
	}
	
	/// \brief Returns the current batch of samples of the Markov chain. 
	SampleBatch& samples(){
		return m_temperedChains;
	}

	///\brief Initializes the markov chain using samples drawn uniformly from the set.
	///
	/// Be aware that the number of chains and the temperatures need to bee specified previously.
	/// @param dataSet the data set
	void initializeChain(Data<RealVector> const& dataSet){
		if(m_temperedChains.size()==0) 
			throw SHARKEXCEPTION("you did not initialize the number of temperatures bevor initializing the chain!");
		DiscreteUniform<typename RBM::RngType> uni(m_operator.rbm()->rng(),0,dataSet.numberOfElements()-1);
		std::size_t visibles = m_operator.rbm()->numberOfVN();
		RealMatrix sampleData(m_temperedChains.size(),visibles);
		
		for(std::size_t i = 0; i != m_temperedChains.size(); ++i){
			noalias(row(sampleData,i)) = dataSet.element(uni());
		}
		initializeChain(sampleData);
	}
	
	/// \brief Initializes with data points from a batch of points
	///
	/// Be aware that the number of chains and the temperatures need to bee specified previously.
	/// @param sampleData the data set
	void initializeChain(RealMatrix const& sampleData){
 		if(m_temperedChains.size()==0) 
			throw SHARKEXCEPTION("you did not initialize the number of temperatures bevor initializing the chain!");

		m_operator.createSample(m_temperedChains.hidden,m_temperedChains.visible,sampleData,m_betas);
		
		m_temperedChains.energy = m_operator.calculateEnergy(
			m_temperedChains.hidden,
			m_temperedChains.visible 
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
				metropolisSwap(
					reference(m_temperedChains,i),m_betas(i),
					reference(m_temperedChains,i+1),m_betas(i+1)
				);
			}
			//ODD phase
			for(std::size_t i = 1; i < elems-1; i+=2){
				metropolisSwap(
					reference(m_temperedChains,i),m_betas(i),
					reference(m_temperedChains,i+1),m_betas(i+1)
				);
			}
			m_operator.rbm()->hiddenNeurons().sufficientStatistics(
				m_temperedChains.hidden.input,m_temperedChains.hidden.statistics, m_betas
			);
		}
	}
};
	
}
#endif
