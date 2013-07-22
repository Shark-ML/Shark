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
#ifndef SHARK_UNSUPERVISED_RBM_SAMPLING_GIBBSOPERATOR_H
#define SHARK_UNSUPERVISED_RBM_SAMPLING_GIBBSOPERATOR_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/IConfigurable.h>
#include "Impl/SampleTypes.h"
namespace shark{
	
///\brief Implements Gibbs Sampling related transition operators for various temperatures.
///
///The operator generates transitions from the current state of the neurons of an RBM 
/// to a new one and thus can be used to produce a Markov chain.
///The Gibbs Operator works by computing the conditional distribution of the hidden given the visible p(h|v) (or
///vice versa) and than samples the new hidden (or visible) state from it.
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
	///Aside fromt the hidden state, this structure can also hold the actual values 
	///of the input, the phi-function and the sufficient statistics
	typedef typename Batch<HiddenSample>::type HiddenSampleBatch;

	///\brief Represents the state of the visible units and additional information required by the gradient.
	///
	///Aside fromt the visible state, this structure can also hold the actual values 
	///of the hidden input, the phi-function and the sufficient statistics
	typedef typename Batch<VisibleSample>::type VisibleSampleBatch;

	///\brief Constructs the Operator using an allready defined Distribution to sample from. 
	GibbsOperator(RBM* rbm):mpe_rbm(rbm){}

	void configure(PropertyTree const& node){}
		
	///\brief Returns the internal RBM.
	RBM* rbm()const{
		return mpe_rbm;
	}


	///\brief Calculates internal data needed for sampling the hidden units as well as requested information for the gradient.
	///
	///This function calculates the conditional propability distribution p(h|v) with inverse temperature beta for the whole batch of samples
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
	///\brief Calculates internal data needed for sampling the hidden units as well as requested information for the gradient.
	///
	///This function calculates the conditional propability distribution p(h|v) with inverse temperature 1 for the whole batch of samples
	///Be aware that a change of temperature may occur between sampleVisible and precomputeHidden.
	void precomputeHidden(HiddenSampleBatch& hiddenBatch, VisibleSampleBatch& visibleBatch)const{
		SIZE_CHECK(visibleBatch.size()==hiddenBatch.size());
		precomputeHidden(hiddenBatch,visibleBatch,RealScalarVector(visibleBatch.size(),1.0));
	}


	///\brief calculates internal data needed for sampling the visible units as well as requested information for the gradient 
	///
	///This function calculates the conditional propability distribution p(v|h) with inverse temperature beta for a whole batch of inputs.
	///Be aware that a change of temperature may occur between sampleHidden and precomputeVisible.
	template<class BetaVector>
	void precomputeVisible(HiddenSampleBatch& hiddenBatch, VisibleSampleBatch& visibleBatch, BetaVector const& beta)const{
		SIZE_CHECK(visibleBatch.size()==hiddenBatch.size());
		mpe_rbm->energy().inputVisible(visibleBatch.input, hiddenBatch.state);
		//calculate the sufficient statistics of the visible units for every element of the batch
		mpe_rbm->visibleNeurons().sufficientStatistics(visibleBatch.input,visibleBatch.statistics, beta);
		
	}
	
	///\brief calculates internal data needed for sampling the visible units as well as requested information for the gradient 
	///
	///This function calculates the conditional propability distribution p(v|h) with inverse temperature beta for a whole batch of inputs.
	///Be aware that a change of temperature may occur between sampleHidden and precomputeVisible.
	void precomputeVisible(HiddenSampleBatch& hiddenBatch, VisibleSampleBatch& visibleBatch)const{
		SIZE_CHECK(visibleBatch.size()==hiddenBatch.size());
		precomputeVisible(hiddenBatch,visibleBatch,RealScalarVector(visibleBatch.size(),1.0));
	}
	

	///\brief Samples a new batch of states of the hidden units using their precomputed statistics.
	void sampleHidden(HiddenSampleBatch& sampleBatch)const{
		//sample state of the hidden neurons, input and statistics was allready computed by precompute
		mpe_rbm->hiddenNeurons().sample(sampleBatch.statistics, sampleBatch.state, mpe_rbm->rng());
	}


	///\brief Samples a new batch of states of the visible units using their precomputed statistics.
	void sampleVisible(VisibleSampleBatch& sampleBatch)const{
		//sample state of the visible neurons, input and statistics was allready computed by precompute
		mpe_rbm->visibleNeurons().sample(sampleBatch.statistics, sampleBatch.state, mpe_rbm->rng());
	}
	

	///\brief Updates the sufficient statistics of the sample after a change of temeprature.
	template<class BetaVector>
	void updateVisible(VisibleSampleBatch& visibleBatch, BetaVector const& newBeta)const{
		mpe_rbm->visibleNeurons().sufficientStatistics(visibleBatch.input,visibleBatch.statistics, newBeta);
	}


	///\brief Updates the value of the sufficient statistics of the sample after a change of temperature.
	template<class BetaVector>
	void updateHidden(HiddenSampleBatch& hiddenBatch, BetaVector const& newBeta)const{
		mpe_rbm->hiddenNeurons().sufficientStatistics(hiddenBatch.input,hiddenBatch.statistics, newBeta);
	}


	///\brief Creates  hidden/visible sample pairs from the states of the visible neurons, i.e. sats the visible units to the given states and samples hidden states based on the states of the visible units. 
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
	
	///\brief Creates  hidden/visible sample pairs from the states of the visible neurons, i.e. sats the visible units to the given states and samples hidden states based on the states of the visible units. 
	/// This can directly be used to calculate the gradient.
	///
	/// @param hiddenBatch the batch of hidden samples to be created
	/// @param visibleBatch the batch of visible samples to be created
	/// @param states the states of the visible neurons in the sample
	/// @param state the state of the visible neurons in the sample
	template<class States>
	void createSample(HiddenSampleBatch& hiddenBatch,VisibleSampleBatch& visibleBatch, States const& states)const{
		createSample(hiddenBatch,visibleBatch,states, RealScalarVector(states.size1(),1.0));
	}
	
	///\brief Calculates the Energy of a sample of the visible and hidden neurons created by this chain.
	/// 
	///@param hiddenBatch the batch of samples of the hidden neurons 
	///@param visibleBatch the batch of samples of the visible neurons (holding also the precomputed input of the visibles)
	///@return the value of the energy function 
	RealVector calculateEnergy(HiddenSampleBatch const& hiddenBatch, VisibleSampleBatch const& visibleBatch)const{
		return mpe_rbm->energy().energyFromVisibleInput(
			visibleBatch.input, 
			hiddenBatch.state, 
			visibleBatch.state
		);
	}
private:
	RBM* mpe_rbm;
};

	
}
#endif
