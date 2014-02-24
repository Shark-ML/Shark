/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_UNSUPERVISED_RBM_SAMPLING_MARKOVCHAIN_H
#define SHARK_UNSUPERVISED_RBM_SAMPLING_MARKOVCHAIN_H

#include <shark/Data/Dataset.h>
#include <shark/Rng/DiscreteUniform.h>
#include <shark/Unsupervised/RBM/Tags.h>
#include <shark/Core/IConfigurable.h>
#include "Impl/SampleTypes.h"
namespace shark{

/// \brief A single Markov chain.
///
/// You can run the Markov chain for some sampling steps by applying a transition operator.
template<class Operator>
class MarkovChain{
private:
	typedef typename Operator::HiddenSample HiddenSample;
	typedef typename Operator::VisibleSample VisibleSample;
public:

	///\brief The MarkovChain can be used to compute several samples at once.
	static const bool computesBatch = true;

	///\brief The type of the RBM the operator is working with.
	typedef typename Operator::RBM RBM;
	///\brief A batch of samples containing hidden and visible samples as well as the energies.
	typedef typename Batch<detail::MarkovChainSample<HiddenSample,VisibleSample> >::type SampleBatch;
	
	///\brief Mutable reference to an element of the batch.
	typedef typename SampleBatch::reference reference;
	
	///\brief Immutable reference to an element of the batch.
	typedef typename SampleBatch::const_reference const_reference;
private:
	///\brief The batch of samples containing the state of the visible and the hidden units. 
	SampleBatch m_samples;   
	///\brief The transition operator.
	Operator m_operator; 
public:

	/// \brief Constructor. 	
	MarkovChain(RBM* rbm):m_operator(rbm){}

		
	/// \brief Sets the number of parallel samples to be evaluated
	void setBatchSize(std::size_t batchSize){
		std::size_t visibles=m_operator.rbm()->numberOfVN();
		std::size_t hiddens=m_operator.rbm()->numberOfHN();
		m_samples=SampleBatch(batchSize,visibles,hiddens);
	}
	std::size_t batchSize(){
		return m_samples.size();
	}
	void configure(PropertyTree const& node){
		m_operator.configure(node);
	}
	
	/// \brief Initializes with data points drawn uniform from the set.
	///
	/// @param dataSet the data set
	void initializeChain(Data<RealVector> const& dataSet){
		DiscreteUniform<typename RBM::RngType> uni(m_operator.rbm()->rng(),0,dataSet.numberOfElements()-1);
		std::size_t visibles=m_operator.rbm()->numberOfVN();
		RealMatrix sampleData(m_samples.size(),visibles);
		
		for(std::size_t i = 0; i != m_samples.size(); ++i){
			noalias(row(sampleData,i)) = dataSet.element(uni());
		}
		initializeChain(sampleData);
	}
	
	/// \brief Initializes with data points from a batch of points
	///
	/// @param sampleData Data set
	void initializeChain(RealMatrix const& sampleData){
		m_operator.createSample(m_samples.hidden,m_samples.visible,sampleData);
	}
	
	/// \brief Runs the chain for a given number of steps.
	/// 
 	/// @param numberOfSteps the number of steps
	void step(unsigned int numberOfSteps){
		m_operator.stepVH(m_samples.hidden,m_samples.visible,numberOfSteps,blas::repeat(1.0,batchSize()));
	}
	
	/// \brief Returns the current sample of the Markov chain. 
	const_reference sample()const{
		return const_reference(m_samples,0);
	}
	
	/// \brief Returns the current batch of samples of the Markov chain. 
	SampleBatch const& samples()const{
		return m_samples;
	}
	
	/// \brief Returns the current batch of samples of the Markov chain. 
	SampleBatch& samples(){
		return m_samples;
	}
	
	/// \brief Returns the transition operator of the Markov chain.
	Operator const& transitionOperator()const{
		return m_operator;
	}

	/// \brief Returns the transition operator of the Markov chain.
	Operator& transitionOperator(){
		return m_operator;
	}
};
	
}
#endif
