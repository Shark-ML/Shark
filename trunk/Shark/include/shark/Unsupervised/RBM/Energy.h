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
#ifndef SHARK_UNSUPERVISED_RBm_ENERGY_H
#define SHARK_UNSUPERVISED_RBm_ENERGY_H

#include <shark/LinAlg/Base.h>

namespace shark{

/// \brief The Energy function determining the Gibbs distribution of an RBM.
///
///General Energy function which uses the information given by the neurons to automatize
///the calculation of the value of the energy for certain states, the derivative of the energy
///and the factorization of the probability.
///
/// Following (but slightly simplifying from the formulas given by) 
/// Welling at al.  a general form of an RBM's Energy function is given by 
/// \f$ E(\vec v,\vec h)=  f_h(\vec h) + f_v(\vec v) +  \sum_{k,l} \phi_{hk}(\vec h) W_{k,l} \phi_{vl}(\vec v) \f$
/// We call \f$ f_h(\vec h) \f$ and  \f$ f_v(\vec v) \f$ the term of the Energy (energy term) 
/// associated to the hidden or the visible neurons respectively.
/// \f$  \sum_{k,l} \phi_{hk}(\vec h) W_{k,l} \phi_{vl}(\vec v) \f$ is called the interaction term.
/// In the standard case of an binary RBM we have \f$ f_h(\vec h) = \vec h  \vec c \f$
/// and \f$ f_v(\vec v) = \vec v \vec b \f$, where \f$ \vec c \f$ and \f$ \vec b \f$
/// are the vectors of the bias parameters for the hidden and the visible neurons respectively.
/// Furthermore, the interaction term simplifies to \f$ \vec h W \vec v \f$, so we have just
/// one singe 'phi-function' for each layer that is the identity function. 
	
template<class RBM>
struct Energy{
	typedef typename RBM::HiddenType HiddenType; //< type of the hidden layer
	typedef typename RBM::VisibleType VisibleType; //< type of the visible layer
	
	//typedefs for single element
	typedef typename HiddenType::SufficientStatistics HiddenStatistics;
	typedef typename VisibleType::SufficientStatistics VisibleStatistics;
	
	//batch typedefs
	typedef typename HiddenType::StatisticsBatch HiddenStatisticsBatch;
	typedef typename VisibleType::StatisticsBatch VisibleStatisticsBatch;

	Energy(VisibleType const& visible, HiddenType const& hidden, RealMatrix const& weightMatrix)
	: m_hiddenNeurons(hidden), m_visibleNeurons(visible), m_weightMatrix(weightMatrix){}
		
	///\brief Calculates the Energy given the states of batches of hidden and visible variables .
	RealVector energy(RealMatrix const& hidden, RealMatrix const& visible)const{
		SIZE_CHECK(size(visible)==size(hidden));
		
		std::size_t batchSize = size(visible);
		RealMatrix input(batchSize,m_hiddenNeurons.size());
		inputHidden( input, visible);
		
		return energyFromHiddenInput( input, hidden, visible);
	}
	
	///\brief Computes the logarithm of the unnormalized probability of each state of the
	/// hidden neurons in a batch by using the precomputed input/activation of the visible neurons.
	///
	///@param hiddenState the batch of states of the hidden neurons
	///@param visibleInput the bacth of current inputs for he visible units given hiddenState
	///@param beta the inverse temperature
	///@return the unnormalized probability
	template<class BetaVector>
	RealVector logUnnormalizedPropabilityHidden(
		RealMatrix const& hiddenState, 
		RealMatrix const& visibleInput, 
		BetaVector beta
	)const{
		SIZE_CHECK(hiddenState.size1()==visibleInput.size1());
		SIZE_CHECK(hiddenState.size1()==beta.size());
		std::size_t batchSize = size(hiddenState);
		
		//calculate the energy terms of the hidden neurons for the whole batch
		RealVector energyTerms = m_hiddenNeurons.energyTerm(hiddenState);

		//calculate resulting probabilities in sequence
		RealVector p(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			double hiddenEnergy = beta(i) * energyTerms(i);
			p(i) = m_visibleNeurons.logMarginalize(row(visibleInput,i),beta(i))+hiddenEnergy;
		}
		return p;
	}


	///\brief Computes the logarithm of the unnormalized probability of each state of the 
	/// visible neurons in a batch by using the precomputed input/activation of the hidden neurons.
	///
	///@param visibleState the batch of states of the hidden neurons
	///@param hiddenInput the batch of current inputs for he visible units given visibleState
	///@param beta the inverse temperature
	///@return the unnormalized probability
	template<class BetaVector>
	RealVector logUnnormalizedPropabilityVisible(
		RealMatrix const& visibleState,
		RealMatrix const& hiddenInput, 
		BetaVector const& beta
	)const{
		SIZE_CHECK(visibleState.size1()==hiddenInput.size1());
		SIZE_CHECK(visibleState.size1()==beta.size());
		std::size_t batchSize = size(visibleState);
		
		//calculate the energy terms of the visible neurons for the whole batch
		RealVector energyTerms = m_visibleNeurons.energyTerm(visibleState);
		
		RealVector p(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			double visibleEnergy = beta(i) * energyTerms(i);
			p(i) = m_hiddenNeurons.logMarginalize(row(hiddenInput,i),beta(i))+visibleEnergy;
		}
		return p;
	}

	
	///\brief Computes the logarithm of the unnormalized probability for each state of the visible neurons from a batch.
	///
	///@param visibleStates the batch of states of the hidden neurons
	///@param beta the inverse temperature
	template<class BetaVector>
	RealVector logUnnormalizedPropabilityVisible(RealMatrix const& visibleStates, BetaVector const& beta)const{
		SIZE_CHECK(visibleStates.size1() == beta.size());
		
		RealMatrix hiddenInputs(beta.size(),m_hiddenNeurons.size());
		inputHidden(hiddenInputs,visibleStates);
		return logUnnormalizedPropabilityVisible(visibleStates, hiddenInputs, beta);
	}
	
	///\brief Computes the logarithm of the unnormalized probability of each state of the hidden neurons from a batch.
	///
	///@param hiddenStates a bacth of states of the hidden neurons
	///@param beta the inverse temperature
	template<class BetaVector>
	RealVector logUnnormalizedPropabilityHidden(RealMatrix const& hiddenStates, BetaVector const& beta)const{
		SIZE_CHECK(hiddenStates.size1() == beta.size());
		
		RealMatrix visibleInputs(beta.size(),m_visibleNeurons.size());
		inputVisible(visibleInputs,hiddenStates);
		return logUnnormalizedPropabilityHidden(hiddenStates, visibleInputs, beta);
	}
    
    
	///\brief Calculates the input of the hidden neurons given the state of the visible in a batch-vise fassion.
	///
	///@param inputs the batch of vectors the input of the hidden neurons is stored in
	///@param visibleStates the batch of states of the visible neurons
	void inputHidden(RealMatrix& inputs, RealMatrix const& visibleStates)const{
		SIZE_CHECK(visibleStates.size1() == inputs.size1());
		SIZE_CHECK(inputs.size2() == m_hiddenNeurons.size());
		
		fast_prod(m_visibleNeurons.phi(visibleStates),trans(m_weightMatrix),inputs);
	}


	///\brief Calculates the input of the visible neurons given the state of the hidden.
	///
	///@param inputs the vector the input of the visible neurons is stored in
	///@param hiddenStates the state of the hidden neurons
	void inputVisible(RealMatrix& inputs, RealMatrix const& hiddenStates)const{
		SIZE_CHECK(hiddenStates.size1() == inputs.size1());
		SIZE_CHECK(inputs.size2() == m_visibleNeurons.size());
		
		fast_prod(m_hiddenNeurons.phi(hiddenStates),m_weightMatrix,inputs);
	}

	///\brief Optimization of the calculation of the energy, when the input of the hidden units
	/// and the value of the phi-function of the hidden neurons is already available.
	///
	///@param hiddenInput the vector of inputs of the hidden neurons
	///@param hidden the states of the hidden neurons
 	///@param visible the states of the visible neurons
	///@param phiOfH the values of the phi-function of the hidden neurons
	///@return the value of the energy function
	RealVector energyFromHiddenInput(
		RealMatrix const& hiddenInput,
		RealMatrix const& hidden, 
		RealMatrix const& visible,
		RealMatrix const& phiOfH
	)const{
		std::size_t batchSize = size(hiddenInput);
		RealVector energies(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			energies(i) = -inner_prod(row(hiddenInput,i),row(phiOfH,i));
		}
		energies -= m_hiddenNeurons.energyTerm(hidden);
		energies -= m_visibleNeurons.energyTerm(visible);
		return energies;
	}


	///\brief Optimization of the calculation of the energy, when the input of the visible units.
	/// and the value of the phi-function of the visible neurons is already available.
	///
	///@param visibleInput the vector of inputs of the visible neurons
	///@param hidden the states of the hidden neurons
 	///@param visible the states of the visible neurons
	///@param phiOfV the values of the phi-function of the visible neurons
	///@return the value of the energy function
	RealVector energyFromVisibleInput(
		RealMatrix const& visibleInput,
		RealMatrix const& hidden, 
		RealMatrix const& visible,
		RealMatrix const& phiOfV
	)const{
		std::size_t batchSize = size(visibleInput);
		RealVector energies(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			energies(i) = -inner_prod(row(phiOfV,i),row(visibleInput,i));
		}
		energies -= m_hiddenNeurons.energyTerm(hidden);
		energies -= m_visibleNeurons.energyTerm(visible);
		return energies;
	}


	///\brief Optimization of the calculation of the energy, when the input of the hidden units is already available.
	///@param hiddenInput the vector of inputs of the hidden neurons
	///@param hidden the states of the hidden neurons
 	///@param visible the states of the visible neurons
	///@return the value of the energy function
	RealVector energyFromHiddenInput(
		RealMatrix const& hiddenInput,
		RealMatrix const& hidden, 
		RealMatrix const& visible
	)const{
		return energyFromHiddenInput(hiddenInput,hidden,visible,m_hiddenNeurons.phi(hidden));
	}


	///\brief Optimization of the calculation of the energy, when the input of the visible units is already available.
	///@param visibleInput the vector of inputs of the visible neurons
	///@param hidden the states of the hidden neurons
 	///@param visible the states of the visible neurons
	///@return the value of the energy function
	RealVector energyFromVisibleInput(
		RealMatrix const& visibleInput,
		RealMatrix const& hidden, 
		RealMatrix const& visible
	)const{
 		return energyFromVisibleInput(visibleInput,hidden,visible,m_visibleNeurons.phi(visible));
	}
private:
	HiddenType const& m_hiddenNeurons;
	VisibleType const& m_visibleNeurons;
	RealMatrix const& m_weightMatrix;
	
};

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
	m_logWeightSum(-std::numeric_limits<double>::infinity()){
		SHARK_CHECK(mpe_rbm != 0, "rbm is not allowed to be 0");
		std::size_t const hiddens = mpe_rbm->numberOfHN();
		std::size_t const visibles = mpe_rbm->numberOfVN();
		std::size_t const hiddenParameters = mpe_rbm->hiddenNeurons().numberOfParameters();
		std::size_t const visibleParameters = mpe_rbm->visibleNeurons().numberOfParameters();
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
		SIZE_CHECK(logWeights.size() == shark::size(hiddens));
		SIZE_CHECK(logWeights.size() == shark::size(visibles));
		
		///update the internal state and get the transformed weights for the batch
		RealVector weights = updateWeights(logWeights);
		if(weights.empty()) return;//weights are not relevant to the gradient
		
		std::size_t batchSize = shark::size(hiddens);
		
		//update the gradient
		RealMatrix weightedFeatures = mpe_rbm->visibleNeurons().phi(visibles.state);
		for(std::size_t i = 0; i != batchSize; ++i){
			row(weightedFeatures,i)*= weights(i);
		}
		fast_prod(trans(mpe_rbm->hiddenNeurons().expectedPhiValue(hiddens.statistics)),weightedFeatures,m_deltaWeights,1.0);
		mpe_rbm->visibleNeurons().parameterDerivative(m_deltaBiasVisible,visibles,weights);
		mpe_rbm->hiddenNeurons().expectedParameterDerivative(m_deltaBiasHidden,hiddens,weights);
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
		SIZE_CHECK(logWeights.size() == shark::size(hiddens));
		SIZE_CHECK(logWeights.size() == shark::size(visibles));
		
		///update the internal state and get the transformed weights for the batch
		RealVector weights = updateWeights(logWeights);
		if(weights.empty()) return;
		
		std::size_t batchSize = shark::size(hiddens);
		
		//update the gradient
		RealMatrix weightedFeatures = mpe_rbm->hiddenNeurons().phi(hiddens.state);
		for(std::size_t i = 0; i != batchSize; ++i){
			row(weightedFeatures,i)*= weights(i);
		}
		
		fast_prod(trans(weightedFeatures),mpe_rbm->visibleNeurons().expectedPhiValue(visibles.statistics),m_deltaWeights,1.0);
		mpe_rbm->hiddenNeurons().parameterDerivative(m_deltaBiasHidden,hiddens,weights);
		mpe_rbm->visibleNeurons().expectedParameterDerivative(m_deltaBiasVisible,visibles,weights);
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
		RealVector derivative(mpe_rbm->numberOfParameters());
		init(derivative) << toVector(m_deltaWeights),m_deltaBiasHidden,m_deltaBiasVisible;
		return derivative;
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
	RBM const* mpe_rbm; //structure of the corresponding RBM
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

#endif
