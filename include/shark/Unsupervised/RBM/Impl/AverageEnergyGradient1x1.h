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
#ifndef SHARK_UNSUPERVISED_RBM_AVERAGEENERGYGRADIENT1X1_H
#define SHARK_UNSUPERVISED_RBM_AVERAGEENERGYGRADIENT1X1_H

#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>
#include <shark/LinAlg/BLAS/Initialize.h>
#include <shark/Unsupervised/RBM/Tags.h>

namespace shark{
namespace detail{

///\brief The gradient of the energy averaged over a set of cumulative added samples.
///
/// It is needed by log-likelihood gradient approximators because it delivers the information
/// how the derivatives of certain energy functions look like.  
///
///This is the special case for Neurons with one interaction term only.
template<class Structure>
class AverageEnergyGradient1x1{
public:	
	AverageEnergyGradient1x1(Structure* structure)
	:mpe_structure(structure),
	m_logWeightSum(-std::numeric_limits<double>::infinity()){
		SHARK_CHECK(mpe_structure != 0, "structure is not allowed to be 0");
		std::size_t const hiddens = mpe_structure->weightMatrix(0,0).size1();
		std::size_t const visibles = mpe_structure->weightMatrix(0,0).size2();
		std::size_t const hiddenParameters = mpe_structure->hiddenNeurons().numberOfParameters();
		std::size_t const visibleParameters = mpe_structure->visibleNeurons().numberOfParameters();
		m_deltaWeights.resize(hiddens,visibles);
		m_deltaBiasHidden.resize(hiddenParameters);
		m_deltaBiasVisible.resize(visibleParameters);
	}
	
	///\brief Calculates the weighted expectation of the energy gradient with respect to p(h|v) for a complete Batch.
	///
	///for numerical stability, the logarithm of the weights is used
	///
	///This method weights the member sof the batch using the weights in the thirs parameter
	///@param hiddens a batch of samples for the hidden layer
	///@param visibles a batch of samples of the visible layer
	///@param logWeights the logarithm of the weights for every sample
	template<class HiddenSampleBatch, class VisibleSampleBatch, class WeightVector>
	void addVH(HiddenSampleBatch const& hiddens, VisibleSampleBatch const& visibles, WeightVector const& logWeights){
		SIZE_CHECK(shark::size(logWeights) == shark::size(hiddens));
		SIZE_CHECK(shark::size(logWeights) == shark::size(visibles));
		
		///update the internal state and get the transformed weights for the batch
		RealVector weights = updateWeights(logWeights);
		if(weights.empty()) return;//weights are not relevant to the gradient
		
		std::size_t batchSize = shark::size(hiddens);
		
		//update the gradient
		RealMatrix weightedFeatures(batchSize, m_deltaWeights.size2());
		for(std::size_t i = 0; i != batchSize; ++i){
			noalias(row(weightedFeatures,i))=weights(i)*get(visibles.features,i);
		}
		fast_prod(trans(mpe_structure->hiddenNeurons().expectedPhiValue(hiddens.statistics)),weightedFeatures,m_deltaWeights,1.0);
		mpe_structure->visibleNeurons().parameterDerivative(m_deltaBiasVisible,visibles,weights);
		mpe_structure->hiddenNeurons().expectedParameterDerivative(m_deltaBiasHidden,hiddens,weights);
	}

	///\brief Calculates the weighted expectation of the energy gradient with respect to p(v|h) for a complete Batch.
	///
	///for numerical stability, the logarithm of the weights is used
	///
	///This method weights the member sof the batch using the weights in the thirs parameter
	///@param hiddens a batch of samples for the hidden layer
	///@param visibles a batch of samples of the visible layer
	///@param logWeights the logarithm of the weights for every sample
	template<class HiddenSampleBatch, class VisibleSampleBatch, class WeightVector>
	void addHV(HiddenSampleBatch const& hiddens, VisibleSampleBatch const& visibles, WeightVector const& logWeights){
		SIZE_CHECK(shark::size(logWeights) == shark::size(hiddens));
		SIZE_CHECK(shark::size(logWeights) == shark::size(visibles));
		
		///update the internal state and get the transformed weights for the batch
		RealVector weights = updateWeights(logWeights);
		if(weights.empty()) return;
		
		std::size_t batchSize = shark::size(hiddens);
		
		//update the gradient
		RealMatrix weightedFeatures(batchSize, m_deltaWeights.size1());
		for(std::size_t i = 0; i != batchSize; ++i){
			noalias(row(weightedFeatures,i))=weights(i)*get(hiddens.features,i);
		}
		
		fast_prod(trans(weightedFeatures),mpe_structure->visibleNeurons().expectedPhiValue(visibles.statistics),m_deltaWeights,1.0);
		mpe_structure->hiddenNeurons().parameterDerivative(m_deltaBiasHidden,hiddens,weights);
		mpe_structure->visibleNeurons().expectedParameterDerivative(m_deltaBiasVisible,visibles,weights);
	}
	
	///\brief Calculates the expectation of the energy gradient with respect to p(h|v) for a complete Batch.
	///
	///for numerical stability, the logarithm of the weights is used
	///
	///This method weights the member sof the batch using the weights in the thirs parameter
	///@param hiddens a batch of samples for the hidden layer
	///@param visibles a batch of samples of the visible layer
	template<class HiddenSampleBatch, class VisibleSampleBatch>
	void addVH(HiddenSampleBatch const& hiddens, VisibleSampleBatch const& visibles){
		addVH(hiddens,visibles, RealScalarVector(shark::size(hiddens),0.0));
	}

	///\brief Calculates the weighted expectation of the energy gradient with respect to p(v|h) for a complete Batch.
	///
	///for numerical stability, the logarithm of the weights is used
	///
	///This method weights the member sof the batch using the weights in the thirs parameter
	///@param hiddens a batch of samples for the hidden layer
	///@param visibles a batch of samples of the visible layer
	template<class HiddenSampleBatch, class VisibleSampleBatch>
	void addHV(HiddenSampleBatch const& hiddens, VisibleSampleBatch const& visibles){
		addHV(hiddens,visibles, RealScalarVector(shark::size(hiddens),0.0));
	}
	
	///Returns the log of the sum of the weights.
	///
	///@return the logarithm of the sum of weights
	double logWeightSum(){
		return m_logWeightSum;
	}

	///\brief Writes the derivatives of all parameters into a vector and returns it.
	RealVector result()const{
		RealVector derivative(mpe_structure->numberOfParameters());
		init(derivative) << toVector(m_deltaWeights),m_deltaBiasHidden,m_deltaBiasVisible;
		return derivative;
	}
	
	///\brief Returns the flags indicating what the sampler should store for Visible->Hidden Markov chains.
	SamplingFlags flagsVH()const{
		SamplingFlags flags;
		flags |= convertToSamplingFlags(mpe_structure->hiddenNeurons().flagsExpectedGradient(), mpe_structure->visibleNeurons().flagsGradient());
		flags |= StoreVisibleFeatures;
		flags |= StoreHiddenStatistics;
		return flags;
	}

	///\brief Returns the flags indicating what the sampler should store for Hidden->Visible markov Chains.
	SamplingFlags flagsHV()const{
		SamplingFlags flags;
		flags |= convertToSamplingFlags(mpe_structure->hiddenNeurons().flagsGradient(), mpe_structure->visibleNeurons().flagsExpectedGradient());
		flags |= StoreHiddenFeatures;
		flags |= StoreVisibleStatistics;
		return flags;
	}
	
	///\brief Resets the internal state. 
	void clear(){
		m_deltaWeights.clear();
		m_deltaBiasVisible.clear();
		m_deltaBiasHidden.clear();
		m_logWeightSum = -std::numeric_limits<double>::infinity();
	}
	
private:
	RealMatrix m_deltaWeights; //stores the average of the derivatives with respect to the weights
	RealVector m_deltaBiasHidden; //stores the average of the derivative with respect to the hidden biases
	RealVector m_deltaBiasVisible; //stores the average of the derivative with respect to the visible biases
	Structure const* mpe_structure; //structure of the corresponding RBM
	double m_logWeightSum; //log of sum of weights. Usually equal to the log of the number of samples used.
	
	
	template<class WeightVector>
	RealVector updateWeights(WeightVector const& logWeights){
		double const minExp = minExpInput<double>();
		double const maxExp = maxExpInput<double>();
		
		//calculate the gradient update with respect of only the current batch
		std::size_t batchSize = shark::size(logWeights);
		//first calculate the batchLogWeightSum
		double batchLogWeightSum = logWeights(0);
		for(std::size_t i = 1; i != batchSize; ++i){
			double const diff = logWeights(i) - batchLogWeightSum;
			if(diff >= maxExp || diff <= minExp){
				if(logWeights(i) > batchLogWeightSum)
					batchLogWeightSum = logWeights(i);
			}
			else
				batchLogWeightSum += softPlus(diff);
		}
		
		double weightSumDiff = batchLogWeightSum-m_logWeightSum;
		//check whether any new weight is big enough to have an effect
		if(weightSumDiff <= minExp )
			return RealVector();
		
		//if the old weights are to small, there is no use in keeping them
		if(weightSumDiff >= maxExp ){
			clear();
			m_logWeightSum = batchLogWeightSum;
		}
		else
		{
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
		RealVector weights(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			weights(i) = std::exp(logWeights(i)-m_logWeightSum);
		}
		return weights;
	}
};
	
}
}
#endif
