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
#ifndef SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_MULTICHAINAPPROXIMATOR_H
#define SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_MULTICHAINAPPROXIMATOR_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include "Impl/DataEvaluator.h"
#include <vector>

namespace shark{
///\brief Approximates the gradient by taking samples from an ensemble of Markov chains running in parallel.
///
///The advantage is, that every chain can produce samples of a different mode of the distribution.
///The disadvantage is however, that mixing is slower and a higher value of sampling steps between subsequent samples
///need to be chosen. 
template<class MarkovChainType>	
class MultiChainApproximator: public SingleObjectiveFunction{
public:
	typedef typename MarkovChainType::RBM RBM;
	
	MultiChainApproximator(RBM* rbm)
	: mpe_rbm(rbm),m_chainOperator(rbm),m_k(1),m_samples(0),m_numBatches(0),m_regularizer(0){
		SHARK_ASSERT(rbm != NULL);
		setBatchSize(500);

		m_features.reset(HAS_VALUE);
		m_features |=HAS_FIRST_DERIVATIVE;
		m_features |=CAN_PROPOSE_STARTING_POINT;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "MultiChainApproximator"; }
	
	void setK(unsigned int k){
		m_k = k;
	}
	void setNumberOfSamples(std::size_t samples){
		m_samples = samples;
	}
	void setBatchSize(std::size_t batchSize){
		m_batchSize = batchSize;
		if(!MarkovChainType::computesBatch)
			m_batchSize=1;
	}
	
	MarkovChainType& chain(){
		return m_chainOperator;
	}
	MarkovChainType const& chain() const{
		return m_chainOperator;
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
	
	void setData(UnlabeledData<RealVector> const& data){
		m_data = data;
		
		//construct a gradient object to get the information about which values of the samples are needed
		AverageEnergyGradient<RBM> grad(mpe_rbm);
		
		//if the number of samples is 0 = unset, set it to the number of points in the data set
		if(!m_samples){
			setNumberOfSamples(m_data.numberOfElements());
		}
		
		//calculate the number of batches
		std::size_t batches = m_samples / m_batchSize; 
		if(m_samples - batches*m_batchSize != 0){
			++batches;
		}
		m_chains.resize(batches);
		
		//swap every sample batch from the vector into the operator, initialize it and shift it back out.
		for(std::size_t i = 0; i != batches;++i){
			swap(m_chains[i],m_chainOperator.samples());
			std::size_t currentBatchSize = std::min(m_samples-i*m_batchSize, m_batchSize);
			m_chainOperator.setBatchSize(currentBatchSize);
			m_chainOperator.initializeChain(m_data);
			swap(m_chains[i],m_chainOperator.samples());
		}
	}
	
	void proposeStartingPoint(SearchPointType& startingPoint) const{
		startingPoint = mpe_rbm->parameterVector();
	}
	
	std::size_t numberOfVariables()const{
		return mpe_rbm->numberOfParameters();
	}
	
	void setRegularizer(double factor, SingleObjectiveFunction* regularizer){
		m_regularizer = regularizer;
		m_regularizationStrength = factor;
	}
	
	double evalDerivative( SearchPointType const & parameter, FirstOrderDerivative & derivative ) const {
		mpe_rbm->setParameterVector(parameter);
		
		AverageEnergyGradient<RBM> modelAverage(mpe_rbm);
		RealVector empiricalAverage = detail::evaluateData(m_data,*mpe_rbm,m_numBatches);
		
		//approximate the expectation of the energy gradient with respect to the model distribution
		//using samples from the Markov chain
		for(std::size_t i = 0; i != m_chains.size();++i){
			swap(m_chains[i],m_chainOperator.samples());//set the current GibbsChain
			m_chainOperator.step(m_k);//do the next step along the gibbs chain
			modelAverage.addVH(m_chainOperator.samples().hidden, m_chainOperator.samples().visible);//update gradient
			swap(m_chains[i],m_chainOperator.samples());//save the GibbsChain.
		}
		
		derivative.resize(mpe_rbm->numberOfParameters());
		noalias(derivative) = modelAverage.result() - empiricalAverage;
		
		if(m_regularizer){
			FirstOrderDerivative regularizerDerivative;
			m_regularizer->evalDerivative(parameter,regularizerDerivative);
			noalias(derivative) += m_regularizationStrength*regularizerDerivative;
		}
		
		return std::numeric_limits<double>::quiet_NaN();
	}
private:
	RBM* mpe_rbm;
	mutable MarkovChainType m_chainOperator;
	mutable std::vector<typename MarkovChainType::SampleBatch> m_chains;
	UnlabeledData<RealVector> m_data;

	unsigned int m_k;
	unsigned int m_samples;
	std::size_t m_batchSize;
	std::size_t m_numBatches;

	SingleObjectiveFunction* m_regularizer;
	double m_regularizationStrength;
};	
}

#endif

