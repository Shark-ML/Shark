/*
*  \par Copyright (c) 1998-2007:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
*/
#ifndef SHARK_UNSUPERVISED_RBM_SAMPLING_TEMPEREDMARKOVCHAIN_H
#define SHARK_UNSUPERVISED_RBM_SAMPLING_TEMPEREDMARKOVCHAIN_H

#include <shark/Data/Dataset.h>
#include <shark/Rng/DiscreteUniform.h>
#include <shark/Unsupervised/RBM/Tags.h>
#include <shark/Core/IConfigurable.h>
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


	///\brief executes a single sampling step in each Markov chain followed by a swapping step between all even and all odd temperatures.
	void step(){
		//do one step of the tempered the Markov chains at the same time
		m_operator.precomputeHidden(m_temperedChains.hidden, m_temperedChains.visible,m_betas);
		m_operator.sampleHidden(m_temperedChains.hidden);
		m_operator.precomputeVisible(m_temperedChains.hidden, m_temperedChains.visible, m_betas);
		m_operator.sampleVisible(m_temperedChains.visible);
		
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
		
		//after a swap, information might be outdated and must be reevaluated. since precompute is called
		//afterwards, only the allready computed precompute of the other neuron must be reevaluated
// 		for(std::size_t i = 0; i != m_temperedChains.size();++i){
// 			m_operator.updateVisible(m_temperedChains[i].visible, m_betas[i]);
// 		}
		
		m_operator.precomputeHidden(m_temperedChains.hidden, m_temperedChains.visible,m_betas);
	}
	
public:
	TemperedMarkovChain(RBM* rbm):m_operator(rbm){
	}


	void configure(PropertyTree const& node){
		std::size_t temperatures = node.get("temperatures", 1);
		setNumberOfTemperatures(temperatures);
		for(std::size_t i = 0; i != temperatures; ++i){
				double factor = temperatures - 1;
				setBeta(i,1.0 - i/factor);
			}		

		m_operator.configure(node);
	}
	
	const Operator& transitionOperator()const{
		return m_operator;
	}
	Operator& transitionOperator(){
		return m_operator;
	}
	

    //\brief Sets the number of temperatures and initializes the tempered chains accordingly. 
	//
	// @param number of temperatures  
	void setNumberOfTemperatures(std::size_t temperatures){
		std::size_t visibles=m_operator.rbm()->numberOfVN();
		std::size_t hiddens=m_operator.rbm()->numberOfHN();
		m_temperedChains = SampleBatch(temperatures,visibles,hiddens);
		m_betas.resize(temperatures);
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
	/// @param dataSet the data set
	void initializeChain(Data<RealVector> const& dataSet){
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
			step();
		}
	}
	
	//is called after the weights of the rbm got updated. 
	//this allows the chains to store intermediate results
	void update(){
		m_operator.precomputeHidden(m_temperedChains.hidden, m_temperedChains.visible, m_betas);
	}
};
	
}
#endif
