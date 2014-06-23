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
#ifndef SHARK_UNSUPERVISED_RBM_SINGLECHAINAPPROXIMATOR_H
#define SHARK_UNSUPERVISED_RBM_SINGLECHAINAPPROXIMATOR_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include "Impl/DataEvaluator.h"

namespace shark{
	
///\brief Approximates the gradient by taking samples from a single Markov chain.
///
///Taking samples only from a single chain leads to a high mixing rate but the correlation of the samples is higher than using
///several chains. This approximator should be used with a sampling scheme which also achieves a faster decorrelation of samples like
///tempering.
template<class MarkovChainType>	
class SingleChainApproximator: public SingleObjectiveFunction{
public:
	typedef typename MarkovChainType::RBM RBM;
	
	SingleChainApproximator(RBM* rbm)
	: mpe_rbm(rbm),m_chain(rbm),m_k(1)
	,m_samples(0),m_batchSize(500)
	,m_numBatches(0){
		SHARK_ASSERT(rbm != NULL);

		m_features.reset(HAS_VALUE);
		m_features |= HAS_FIRST_DERIVATIVE;
		m_features |= CAN_PROPOSE_STARTING_POINT;
		
		m_chain.setBatchSize(1);
	};

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SingleChainApproximator"; }

	void setK(unsigned int k){
		m_k = k;
	}
	void setNumberOfSamples(std::size_t samples){
		m_samples = samples;
	}
	
	/// \brief Returns the number of batches of the dataset that are used in every iteration.
	///
	/// If it is less than all batches, the batches are chosen at random. if it is 0, all batches are used
	std::size_t numBatches()const{
		return m_numBatches;
	}
	
	/// \brief Returns a reference to the number of batches of the dataset that are used in every iteration.
	///
	/// If it is less than all batches, the batches are chosen at random.if it is 0, all batches are used.
	std::size_t& numBatches(){
		return m_numBatches;
	}
	
	MarkovChainType& chain(){
		return m_chain;
	}
	MarkovChainType const& chain() const{
		return m_chain;
	}
	
	void configure(PropertyTree const& node){
		PropertyTree::const_assoc_iterator it = node.find("rbm");
		if(it!=node.not_found())
		{
			mpe_rbm->configure(it->second);
		}
		setK(node.get<unsigned int>("k",1));
		setNumberOfSamples(node.get<unsigned int>("samples",0));
	}
	
	void setData(UnlabeledData<RealVector> const& data){
		m_data = data;
		m_chain.initializeChain(m_data);
	}

	void proposeStartingPoint(SearchPointType& startingPoint) const{
		startingPoint = mpe_rbm->parameterVector();
	}
	
	std::size_t numberOfVariables()const{
		return mpe_rbm->numberOfParameters();
	}
	
	double evalDerivative( SearchPointType const & parameter, FirstOrderDerivative & derivative ) const {
		mpe_rbm->setParameterVector(parameter);
		
		AverageEnergyGradient<RBM> modelAverage(mpe_rbm);
		RealVector empiricalAverage = detail::evaluateData(m_data,*mpe_rbm,m_numBatches);
		
		//approximate the expectation of the energy gradient with respect to the model distribution
		//using samples from the Markov chain
		
		//calculate number of samples to draw and size of batches used in the gradient update
		std::size_t samplesToDraw = m_samples > 0 ? m_samples: m_data.numberOfElements();
		
		std::size_t batches = samplesToDraw / m_batchSize; 
		if(samplesToDraw - batches*m_batchSize != 0){
			++batches;
		}
		
		//calculate the gradient. we do this by normal k-step sampling for exactly as many
		//samples as calculated in samplesToDraw but saving the result in an intermediate
		//batch variable gradientbatch. When this batch is full, we do an update step of the gradient.
		//this is an a bit more efficient grouping and preserves us from using batches of size1 as the argument 
		//of addVH which might be inefficient.
		for(std::size_t batch = 0; batch != batches; ++batch){
			//calculate the size of the next batch which is batchSize as long as there are enough samples left to draw
			std::size_t currentBatchSize = std::min(samplesToDraw-batch*m_batchSize, m_batchSize);
			typename MarkovChainType::SampleBatch gradientBatch(currentBatchSize, mpe_rbm->numberOfVN(),mpe_rbm->numberOfHN());
			//fill the batch with fresh samples
			for(std::size_t i = 0; i != currentBatchSize; ++i){
				m_chain.step(m_k);
				get(gradientBatch,i) = m_chain.sample();
			}
			//do the gradient update
			modelAverage.addVH(gradientBatch.hidden, gradientBatch.visible);
		}
		
		derivative.resize(mpe_rbm->numberOfParameters());
		noalias(derivative) = modelAverage.result() - empiricalAverage;
	
		return std::numeric_limits<double>::quiet_NaN();
	}

private:
	RBM* mpe_rbm;
	mutable MarkovChainType m_chain; 
	UnlabeledData<RealVector> m_data;

	unsigned int m_k;
	unsigned int m_samples;
	std::size_t m_batchSize;
	std::size_t m_numBatches;
};	
	
}

#endif
