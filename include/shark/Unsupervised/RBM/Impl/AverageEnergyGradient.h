/*!
 * \brief       Implements the gradient of the energy function with respect to its parameters for the RBM
 * 
 * \author      O. Krause
 * \date        2015
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
#ifndef SHARK_UNSUPERVISED_RBM_IMPL_AVERAGEENERGYGRADIENT_H
#define SHARK_UNSUPERVISED_RBM_IMPL_AVERAGEENERGYGRADIENT_H

#include <shark/LinAlg/Base.h>
namespace shark{
namespace detail{
///\brief The gradient of the energy averaged over a set of cumulative added samples.
///
/// It is needed by log-likelihood gradient approximators because it delivers the information
/// how the derivatives of certain energy functions look like.  
///
///This is the special case for Neurons with one interaction term only.
template<class RBM>
class AverageEnergyGradient{
public:	
	AverageEnergyGradient(RBM const* rbm)
	:mpe_rbm(rbm),
	m_logWeightSum(-1e100){
		SHARK_RUNTIME_CHECK(mpe_rbm != 0, "rbm is not allowed to be 0");
		std::size_t const hiddens = mpe_rbm->numberOfHN();
		std::size_t const visibles = mpe_rbm->numberOfVN();
		std::size_t const hiddenParameters = mpe_rbm->hiddenNeurons().numberOfParameters();
		std::size_t const visibleParameters = mpe_rbm->visibleNeurons().numberOfParameters();
		m_deltaWeights.resize(hiddens,visibles);
		m_deltaBiasHidden.resize(hiddenParameters);
		m_deltaBiasVisible.resize(visibleParameters);
		clear();
	}
	
	///\brief Calculates the weighted expectation of the energy gradient with respect to p(h|v) for a complete Batch.
	///
	///for numerical stability, the logarithm of the weights is used
	///
	///This method weights the members of the batch using the weights specified by the corresponding parameter
	///@param hiddens a batch of samples for the hidden layer
	///@param visibles a batch of samples of the visible layer
	///@param logWeights the logarithm of the weights for every sample
	template<class HiddenSampleBatch, class VisibleSampleBatch, class WeightVector>
	void addVH(HiddenSampleBatch const& hiddens, VisibleSampleBatch const& visibles, WeightVector const& logWeights){
		SIZE_CHECK(logWeights.size() == batchSize(hiddens));
		SIZE_CHECK(logWeights.size() == batchSize(visibles));
		
		///update the internal state and get the transformed weights for the batch
		RealVector weights = updateWeights(logWeights);
		if(weights.empty()) return;//weights are not relevant to the gradient
		
		std::size_t size = batchSize(hiddens);
		
		//update the gradient
		RealMatrix weightedFeatures = mpe_rbm->visibleNeurons().phi(visibles.state);
		for(std::size_t i = 0; i != size; ++i){
			row(weightedFeatures,i)*= weights(i);
		}
		noalias(m_deltaWeights) += prod(trans(mpe_rbm->hiddenNeurons().expectedPhiValue(hiddens.statistics)),weightedFeatures);
		mpe_rbm->visibleNeurons().parameterDerivative(m_deltaBiasVisible,visibles,weights);
		mpe_rbm->hiddenNeurons().expectedParameterDerivative(m_deltaBiasHidden,hiddens,weights);
	}

	///\brief Calculates the weighted expectation of the energy gradient with respect to p(v|h) for a complete Batch.
	///
	///for numerical stability, the logarithm of the weights is used
	///
	///This method weights the members of the batch using the weights in the specified by the corresponding parameter
	///@param hiddens a batch of samples for the hidden layer
	///@param visibles a batch of samples of the visible layer
	///@param logWeights the logarithm of the weights for every sample
	template<class HiddenSampleBatch, class VisibleSampleBatch, class WeightVector>
	void addHV(HiddenSampleBatch const& hiddens, VisibleSampleBatch const& visibles, WeightVector const& logWeights){
		SIZE_CHECK(logWeights.size() == batchSize(hiddens));
		SIZE_CHECK(logWeights.size() == batchSize(visibles));
		
		///update the internal state and get the transformed weights for the batch
		RealVector weights = updateWeights(logWeights);
		if(weights.empty()) return;
		
		std::size_t size = batchSize(hiddens);
		
		//update the gradient
		RealMatrix weightedFeatures = mpe_rbm->hiddenNeurons().phi(hiddens.state);
		for(std::size_t i = 0; i != size; ++i){
			row(weightedFeatures,i)*= weights(i);
		}
		
		noalias(m_deltaWeights) += prod(trans(weightedFeatures),mpe_rbm->visibleNeurons().expectedPhiValue(visibles.statistics));
		mpe_rbm->hiddenNeurons().parameterDerivative(m_deltaBiasHidden,hiddens,weights);
		mpe_rbm->visibleNeurons().expectedParameterDerivative(m_deltaBiasVisible,visibles,weights);
	}
	
	AverageEnergyGradient& operator+=(AverageEnergyGradient const& gradient){
		double const minExp = minExpInput<double>();
		double const maxExp = maxExpInput<double>();
		
		double weightSumDiff = gradient.m_logWeightSum-m_logWeightSum;
		//check whether the weight is big enough to have an effect
		if(weightSumDiff <= minExp )
			return *this;
		
		//if the old weights are to small, there is no use in keeping them
		if(weightSumDiff >= maxExp ){
			(*this) = gradient;
			return *this;
		}

		double logWeightSumUpdate = softPlus(weightSumDiff);
		m_logWeightSum += logWeightSumUpdate;

		//scaling factor corrects by multiplying with 
		//Z/(Z+Z_new)=1/(1+exp(logZ_new - logZ))
		double const scalingFactor = std::exp(-logWeightSumUpdate);// factor is <=1
		m_deltaWeights *= scalingFactor;
		m_deltaBiasVisible *= scalingFactor;
		m_deltaBiasHidden *= scalingFactor;
		
		//now add the new gradient with its corrected weight
		double weight = std::exp(gradient.m_logWeightSum-m_logWeightSum);
		noalias(m_deltaWeights) += weight * gradient.m_deltaWeights;
		noalias(m_deltaBiasVisible) += weight * gradient.m_deltaBiasVisible;
		noalias(m_deltaBiasHidden) += weight * gradient.m_deltaBiasHidden;
	}
	
	///\brief Calculates the expectation of the energy gradient with respect to p(h|v) for a complete Batch.
	///
	///for numerical stability, the logarithm of the weights is used
	///
	///This method weights the memberas of the batch using the weights specified by the corresponding parameter
	///@param hiddens a batch of samples for the hidden layer
	///@param visibles a batch of samples of the visible layer
	template<class HiddenSampleBatch, class VisibleSampleBatch>
	void addVH(HiddenSampleBatch const& hiddens, VisibleSampleBatch const& visibles){
		addVH(hiddens,visibles, blas::repeat(0.0,batchSize(hiddens)));
	}

	///\brief Calculates the weighted expectation of the energy gradient with respect to p(v|h) for a complete Batch.
	///
	///for numerical stability, the logarithm of the weights is used
	///
	///This method weights the members of the batch using the weights specified by the corresponding parameter
	///@param hiddens a batch of samples for the hidden layer
	///@param visibles a batch of samples of the visible layer
	template<class HiddenSampleBatch, class VisibleSampleBatch>
	void addHV(HiddenSampleBatch const& hiddens, VisibleSampleBatch const& visibles){
		addHV(hiddens,visibles, blas::repeat(0.0,batchSize(hiddens)));
	}
	
	///Returns the log of the sum of the weights.
	///
	///@return the logarithm of the sum of weights
	double logWeightSum(){
		return m_logWeightSum;
	}

	///\brief Writes the derivatives of all parameters into a vector and returns it.
	RealVector result()const{
		return to_vector(m_deltaWeights) | m_deltaBiasHidden | m_deltaBiasVisible;
	}
	
	///\brief Resets the internal state. 
	void clear(){
		m_deltaWeights.clear();
		m_deltaBiasVisible.clear();
		m_deltaBiasHidden.clear();
		m_logWeightSum = -1e100;
	}
	
private:
	RealMatrix m_deltaWeights; //stores the average of the derivatives with respect to the weights
	RealVector m_deltaBiasHidden; //stores the average of the derivative with respect to the hidden biases
	RealVector m_deltaBiasVisible; //stores the average of the derivative with respect to the visible biases
	RBM const* mpe_rbm; //structure of the corresponding RBM
	double m_logWeightSum; //log of sum of weights. Usually equal to the log of the number of samples used.
	
	
	template<class WeightVector>
	RealVector updateWeights(WeightVector const& logWeights){
		
		//calculate the gradient update with respect of only the current batch
		std::size_t size = batchSize(logWeights);
		//first calculate the batchLogWeightSum
		double batchLogWeightSum = logWeights(0);
		for(std::size_t i = 1; i != size; ++i){
			double const diff = logWeights(i) - batchLogWeightSum;
			batchLogWeightSum += softPlus(diff);
		}
		
		if(m_logWeightSum == -1e100){
			m_logWeightSum = batchLogWeightSum;
		}else{
		
			double weightSumDiff = batchLogWeightSum-m_logWeightSum;
			double weightSumUpdate = softPlus(weightSumDiff);
			m_logWeightSum = m_logWeightSum + weightSumUpdate;

			//scaling factor corrects by multiplying with 
			//Z/(Z+Z_new)=1/(1+exp(logZ_new - logZ))
			double const scalingFactor = std::exp(-weightSumUpdate);// factor is <=1
			m_deltaWeights *= scalingFactor;
			m_deltaBiasVisible *= scalingFactor;
			m_deltaBiasHidden *= scalingFactor;
		}
			
		//now calculate the weights for the elements of the new batch
		return exp(logWeights - m_logWeightSum);
	}
};
}}

#endif
