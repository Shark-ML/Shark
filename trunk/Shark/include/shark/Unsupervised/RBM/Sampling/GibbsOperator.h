/*!
 * 
 *
 * \brief       Implements Block Gibbs Sampling
 *
 * \author    O.Krause
 * \date        2014
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
#ifndef SHARK_UNSUPERVISED_RBM_SAMPLING_GIBBSOPERATOR_H
#define SHARK_UNSUPERVISED_RBM_SAMPLING_GIBBSOPERATOR_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/IConfigurable.h>
#include "Impl/SampleTypes.h"
namespace shark{
	
///\brief Implements Block Gibbs Sampling related transition operators for various temperatures.
///
/// The operator generates transitions from the current state of the neurons of an RBM 
/// to a new one and thus can be used to produce a Markov chain.
/// The Gibbs Operator works by computing the conditional distribution of the hidden given the visible p(h|v) (or
/// vice versa) and than samples the new hidden (or visible) state from it.
///
/// As an interesting twist, this operator can also be used to implement Flip-The-State sampling using two values alpha_visible
/// and alpha_hidden both being between 0 and 1 (inclusively). for alpha_visible=alpha_hidden=0, pure gibbs sampling is performed.
/// if for one of the layers, the value is not 0 a mixture of gibbs and flip-the-state sampling is performed. 1 equals to pure flip-the state
/// sampling.
/// The trick of this sampler is that it takes the previous state into account while sampling. If the current state has a low probability,
/// the sampler jumps deterministically in another state with higher probability. This is counterbalanced by having a higher chance to jump away from
/// this state.
template< class RBMType >
class GibbsOperator:public IConfigurable{
public:
	typedef RBMType RBM;

	///The operator holds a 'sample' of the visible and hidden neurons.
	///Such a sample does not only contain the states of the neurons but all other information
	///needed to approximate the gradient

	///\brief the type of a concrete sample.
	typedef detail::GibbsSample<
		typename RBMType::HiddenType::SufficientStatistics
	> HiddenSample; 
	///\brief the type of a concrete sample.
	typedef detail::GibbsSample<
		typename RBMType::VisibleType::SufficientStatistics
	> VisibleSample; 

	///\brief Represents the state of a batch of hidden samples and additional information required by the gradient.
	///
	///Aside from the hidden state, this structure can also hold the actual values 
	///of the input, the phi-function and the sufficient statistics
	typedef typename Batch<HiddenSample>::type HiddenSampleBatch;

	///\brief Represents the state of the visible units and additional information required by the gradient.
	///
	///Aside from the visible state, this structure can also hold the actual values 
	///of the hidden input, the phi-function and the sufficient statistics
	typedef typename Batch<VisibleSample>::type VisibleSampleBatch;

	///\brief Constructs the Operator using an allready defined Distribution to sample from. 
	GibbsOperator(RBM* rbm, double alphaVisible = 0,double alphaHidden = 0)
	:mpe_rbm(rbm), m_alphaVisible(alphaVisible),m_alphaHidden(alphaHidden){
		SHARK_CHECK(alphaVisible >= 0.0, "alpha >= 0 not fulfilled for the visible layer");
		SHARK_CHECK(alphaVisible <= 1., "alpha <=1 not fulfilled for the visible layer");
		SHARK_CHECK(alphaHidden >= 0.0, "alpha >= 0 not fulfilled for the hidden layer");
		SHARK_CHECK(alphaHidden <= 1., "alpha <=1 not fulfilled for the hidden layer");
	}

	void configure(PropertyTree const& node){}
		
	///\brief Returns the internal RBM.
	RBM* rbm()const{
		return mpe_rbm;
	}


	///\brief Calculates internal data needed for sampling the hidden units as well as requested information for the gradient.
	///
	///This function calculates the conditional probability distribution p(h|v) with inverse temperature beta for the whole batch of samples
	///Be aware that a change of temperature may occur between sampleVisible and precomputeHidden.
	/// @param hiddenBatch the batch of hidden samples to be created
	/// @param visibleBatch the batch of visible samples to be created
	/// @param beta the vector of inverse temperatures
	template<class BetaVector>
	void precomputeHidden(HiddenSampleBatch& hiddenBatch, VisibleSampleBatch& visibleBatch, BetaVector const& beta)const{
		SIZE_CHECK(visibleBatch.size()==hiddenBatch.size());
		mpe_rbm->energy().inputHidden(hiddenBatch.input, visibleBatch.state);
		//calculate the sufficient statistics of the hidden units
		mpe_rbm->hiddenNeurons().sufficientStatistics(hiddenBatch.input,hiddenBatch.statistics, beta);
	}


	///\brief calculates internal data needed for sampling the visible units as well as requested information for the gradient 
	///
	///This function calculates the conditional probability distribution p(v|h) with inverse temperature beta for a whole batch of inputs.
	///Be aware that a change of temperature may occur between sampleHidden and precomputeVisible.
	template<class BetaVector>
	void precomputeVisible(HiddenSampleBatch& hiddenBatch, VisibleSampleBatch& visibleBatch, BetaVector const& beta)const{
		SIZE_CHECK(visibleBatch.size()==hiddenBatch.size());
		mpe_rbm->energy().inputVisible(visibleBatch.input, hiddenBatch.state);
		//calculate the sufficient statistics of the visible units for every element of the batch
		mpe_rbm->visibleNeurons().sufficientStatistics(visibleBatch.input,visibleBatch.statistics, beta);
		
	}

	///\brief Samples a new batch of states of the hidden units using their precomputed statistics.
	void sampleHidden(HiddenSampleBatch& sampleBatch)const{
		//sample state of the hidden neurons, input and statistics was allready computed by precompute
		mpe_rbm->hiddenNeurons().sample(sampleBatch.statistics, sampleBatch.state, m_alphaHidden, mpe_rbm->rng());
	}


	///\brief Samples a new batch of states of the visible units using their precomputed statistics.
	void sampleVisible(VisibleSampleBatch& sampleBatch)const{
		//sample state of the visible neurons, input and statistics was allready computed by precompute
		mpe_rbm->visibleNeurons().sample(sampleBatch.statistics, sampleBatch.state, m_alphaVisible, mpe_rbm->rng());
	}
	
	/// \brief Applies the Gibbs operator a number of times to a given sample.
	///
	/// Performs one complete step for a sample by sampling first the hidden, than the visible and computing the probability of a hidden given the visible unit
	/// That is, Given a State (v,h), computes p(v|h),draws v and then computes p(h|v) and draws h . this is repeated several times
	template<class BetaVector>
	void stepVH(HiddenSampleBatch& hiddenBatch, VisibleSampleBatch& visibleBatch, std::size_t numberOfSteps, BetaVector const& beta){
		for(unsigned int i=0; i != numberOfSteps; i++){
			precomputeVisible(hiddenBatch,visibleBatch,beta);
			sampleVisible(visibleBatch);
			precomputeHidden(hiddenBatch, visibleBatch,beta);
			sampleHidden(hiddenBatch);
		}
	}

	///\brief Creates  hidden/visible sample pairs from the states of the visible neurons, i.e. sets the visible units to the given states and samples hidden states based on the states of the visible units. 
	/// This can directly be used to calculate the gradient.
	///
	/// @param hiddenBatch the batch of hidden samples to be created
	/// @param visibleBatch the batch of visible samples to be created
	/// @param states the states of the visible neurons in the sample
	/// @param beta the vector of inverse temperatures
	template<class States, class BetaVector>
	void createSample(HiddenSampleBatch& hiddenBatch,VisibleSampleBatch& visibleBatch, States const& states, BetaVector const& beta)const{
		SIZE_CHECK(size(states)==visibleBatch.size());
		SIZE_CHECK(hiddenBatch.size()==visibleBatch.size());
		visibleBatch.state = states;
		
		precomputeHidden(hiddenBatch,visibleBatch, beta);
		sampleHidden(hiddenBatch);
	}
	
	///\brief Creates  hidden/visible sample pairs from the states of the visible neurons, i.e. sets the visible units to the given states and samples hidden states based on the states of the visible units. 
	/// This can directly be used to calculate the gradient.
	///
	/// @param hiddenBatch the batch of hidden samples to be created
	/// @param visibleBatch the batch of visible samples to be created
	/// @param states the states of the visible neurons in the sample
	template<class States>
	void createSample(HiddenSampleBatch& hiddenBatch,VisibleSampleBatch& visibleBatch, States const& states)const{
		createSample(hiddenBatch,visibleBatch,states, blas::repeat(1.0,states.size1()));
	}
	
	///\brief Calculates the Energy of a sample of the visible and hidden neurons created by this chain.
	/// 
	///@param hiddenBatch the batch of samples of the hidden neurons 
	///@param visibleBatch the batch of samples of the visible neurons (holding also the precomputed input of the visibles)
	///@return the value of the energy function 
	RealVector calculateEnergy(HiddenSampleBatch const& hiddenBatch, VisibleSampleBatch const& visibleBatch)const{
		return mpe_rbm->energy().energyFromHiddenInput(
			hiddenBatch.input, 
			hiddenBatch.state, 
			visibleBatch.state
		);
	}
private:
	RBM* mpe_rbm;
	double m_alphaVisible;
	double m_alphaHidden;
};

	
}
#endif
