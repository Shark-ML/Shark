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
#ifndef SHARK_UNSUPERVISED_RBM_SAMPLING_ESTEMPEREDMARKOVCHAIN_H
#define SHARK_UNSUPERVISED_RBM_SAMPLING_ESTEMPEREDMARKOVCHAIN_H


#include <shark/Unsupervised/RBM/Sampling/TemperedMarkovChain.h>
namespace shark{
	

///\brief Implements parallel tempering but also stores additional statistics on the energy differences
template<class Operator>
class EnergyStoringTemperedMarkovChain{
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
	typedef typename TemperedMarkovChain<Operator>::SampleBatch SampleBatch;
	
	///\brief Mutable reference to an element of the batch.
	typedef typename SampleBatch::reference reference;
	
	///\brief Immutable reference to an element of the batch.
	typedef typename SampleBatch::const_reference const_reference;
	
private:
	
	TemperedMarkovChain<Operator> m_chain;
	
	bool m_storeEnergyDifferences;
	bool m_integrateEnergyDifferences;
	std::vector<RealVector> m_energyDiffUp;
	std::vector<RealVector> m_energyDiffDown;
	
public:
	EnergyStoringTemperedMarkovChain(RBM* rbm, 
		bool integrateEnergyDifferences = true
	):m_chain(rbm)
	, m_integrateEnergyDifferences(integrateEnergyDifferences)
	, m_storeEnergyDifferences(true){}
	
	const Operator& transitionOperator()const{
		return m_chain.transitionOperator();
	}
	Operator& transitionOperator(){
		return m_chain.transitionOperator();
	}
	
	void setNumberOfTemperatures(std::size_t temperatures){
		m_chain.setNumberOfTemperatures(temperatures);
	}
	void setUniformTemperatureSpacing(std::size_t temperatures){
		m_chain.setUniformTemperatureSpacing(temperatures);
	}

	/// \brief Returns the number Of temperatures.
	std::size_t numberOfTemperatures()const{
		return m_chain.numberOfTemperatures();
	}
	
	void setBatchSize(std::size_t batchSize){
		SHARK_CHECK(batchSize == 1, "[TemperedMarkovChain::setBatchSize] markov chain can only compute batches of size 1.");
	}
	std::size_t batchSize(){
		return 1;
	}
	
	void setBeta(std::size_t i, double beta){
		m_chain.setBeta(i,beta);
	}
	
	double beta(std::size_t i)const{
		return m_chain.beta(i);
	}
	
	RealVector const& beta()const{
		return m_chain.beta();
	}
	
	///\brief Returns the current state of the chain for beta = 1.
	const_reference sample()const{
		return m_chain.sample();
	}
	///\brief Returns the current state of the chain for all beta values.
	SampleBatch const& samples()const{
		return m_chain.samples();
	}
	
	/// \brief Returns the current batch of samples of the Markov chain. 
	SampleBatch& samples(){
		return m_chain.samples();
	}

	///\brief Initializes the markov chain using samples drawn uniformly from the set.
	///
	/// @param dataSet the data set
	void initializeChain(Data<RealVector> const& dataSet){
		m_chain.initializeChain(dataSet);
	}
	
	/// \brief Initializes with data points from a batch of points
	///
	/// @param sampleData the data set
	void initializeChain(RealMatrix const& sampleData){
 		m_chain.initializeChain(sampleData);
	}
	//updates the chain using the current sample
	void step(unsigned int k){
		m_chain.step(k);
		
		if(!storeEnergyDifferences()) return;
		
		typename RBM::EnergyType energy = transitionOperator().rbm()->energy();
		std::size_t numChains = beta().size();
		//create diff beta vectors
		RealVector betaUp(numChains);
		RealVector betaDown(numChains);
		betaUp(0) = 1.0;
		betaDown(numChains-1) = 0.0;
		for(std::size_t i = 0; i != numChains-1; ++i){
			betaDown(i) = beta()(i+1);
			betaUp(i+1) = beta()(i);
		}
		
		RealVector energyDiffUp(numChains);
		RealVector energyDiffDown(numChains);
		if(!m_integrateEnergyDifferences){
			noalias(energyDiffUp) = samples().energy*(betaUp-beta());
			noalias(energyDiffDown) = samples().energy*(betaDown-beta());
		}
		else{
			//calculate the first term: -E(state,beta) thats the same for both matrices
			energy.inputVisible(samples().visible.input, samples().hidden.state);
			noalias(energyDiffDown) = energy.logUnnormalizedProbabilityHidden(
				samples().hidden.state,
				samples().visible.input,
				beta()
			);
			noalias(energyDiffUp) = energyDiffDown;
			
			//now add the new term
			noalias(energyDiffUp) -= energy.logUnnormalizedProbabilityHidden(
				samples().hidden.state,
				samples().visible.input,
				betaUp
			);
			noalias(energyDiffDown) -= energy.logUnnormalizedProbabilityHidden(
				samples().hidden.state,
				samples().visible.input,
				betaDown
			);
		}
		m_energyDiffUp.push_back(energyDiffUp);
		m_energyDiffDown.push_back(energyDiffDown);
	}
	
	RealMatrix getUpDifferences()const{
		RealMatrix diffUp(beta().size(),m_energyDiffUp.size());
		for(std::size_t i = 0; i != m_energyDiffUp.size(); ++i){
			noalias(column(diffUp,i)) = m_energyDiffUp[i];
		}
		return diffUp;
	}
	RealMatrix getDownDifferences()const{
		RealMatrix diffDown(beta().size(),m_energyDiffDown.size());
		for(std::size_t i = 0; i != m_energyDiffDown.size(); ++i){
			noalias(column(diffDown,i)) = m_energyDiffDown[i];
		}
		return diffDown;
	}
	
	void resetDifferences(){
		m_energyDiffUp.clear();
		m_energyDiffDown.clear();
	}
	
	bool& storeEnergyDifferences(){
		return m_storeEnergyDifferences;
	}
	
	//is called after the weights of the rbm got updated. 
	//this allows the chains to store intermediate results
	void update(){
		m_chain.update();
	}
};
	
}
#endif
