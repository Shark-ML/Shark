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

#include <shark/Unsupervised/RBM/Tags.h>
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
	typedef typename RBM::Energy Energy;
	typedef typename Energy::Structure Structure;

	///The operator holds a 'sample' of the visible and hidden neurons.
	///Such a sample does not only contain the states of the neurons but all other information
	///needed to approximate the gradient

	///\brief the type of a concrete sample.
	typedef detail::GibbsSample<
		typename Energy::HiddenInput,
		typename Energy::HiddenStatistics, 
		typename Energy::HiddenFeatures, 
		typename Energy::HiddenState
	> HiddenSample; 
	///\brief the type of a concrete sample.
	typedef detail::GibbsSample<
		typename Energy::VisibleInput,
		typename Energy::VisibleStatistics, 
		typename Energy::VisibleFeatures, 
		typename Energy::VisibleState
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

	void configure(PropertyTree const& node){
	}
		
	///\brief Returns the internal RBM.
	RBM* rbm()const{
		return mpe_rbm;
	}
	
	///\brief Returns the current settings of Flags, which describe which values are to be computed by the Operator.
	SamplingFlags& flags(){
		return m_flags;
	}
	///\brief Returns the current settings of Flags, which describe which values are to be computed by the Operator.
	const SamplingFlags& flags()const{
		return m_flags;
	}


	///\brief Calculates internal data needed for sampling the hidden units as well as requested information for the gradient.
	///
	///This function calculates the conditional propability distribution p(h|v) at temperature beta for the whole batch of samples
	///Additionally, requested information set by flags() is calculated for the hidden part
	///Be aware that a change of temperature may occur between sampleVisible and precomputeHidden.
	template<class BetaVector>
	void precomputeHidden(HiddenSampleBatch& hiddenBatch, VisibleSampleBatch& visibleBatch, BetaVector const& beta)const{
		SIZE_CHECK(visibleBatch.size()==hiddenBatch.size());
		calculateHiddenInput(hiddenBatch,visibleBatch);
		//calculate the sufficient statistics of the hidden units
		mpe_rbm->hiddenNeurons().sufficientStatistics(hiddenBatch.input,hiddenBatch.statistics, beta);
	}
	///\brief Calculates internal data needed for sampling the hidden units as well as requested information for the gradient.
	///
	///This function calculates the conditional propability distribution p(h|v) at temperature 1 for the whole batch of samples
	///Additionally, requested information set by flags() is calculated for the hidden part
	///Be aware that a change of temperature may occur between sampleVisible and precomputeHidden.
	void precomputeHidden(HiddenSampleBatch& hiddenBatch, VisibleSampleBatch& visibleBatch)const{
		SIZE_CHECK(visibleBatch.size()==hiddenBatch.size());
		precomputeHidden(hiddenBatch,visibleBatch,RealScalarVector(visibleBatch.size(),1.0));
	}


	///\brief calculates internal data needed for sampling the visible units as well as requested information for the gradient 
	///
	///This function calculates the conditional propability distribution p(v|h) at temperature beta for a whole batch of inputs.
	///Additionally, requested information set by flags() is calculated for the visible part
	///Be aware that a change of temperature may occur between sampleHidden and precomputeVisible.
	template<class BetaVector>
	void precomputeVisible(HiddenSampleBatch& hiddenBatch, VisibleSampleBatch& visibleBatch, BetaVector const& beta)const{
		SIZE_CHECK(visibleBatch.size()==hiddenBatch.size());
		calculateVisibleInput(hiddenBatch,visibleBatch);
		//calculate the sufficient statistics of the visible units for every element of the batch
		mpe_rbm->visibleNeurons().sufficientStatistics(visibleBatch.input,visibleBatch.statistics, beta);
		
	}
	
	///\brief calculates internal data needed for sampling the visible units as well as requested information for the gradient 
	///
	///This function calculates the conditional propability distribution p(v|h) at temperature beta for a whole batch of inputs.
	///Additionally, requested information set by flags() is calculated for the visible part
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
	///
	///After changing the temperature, requested information regarding the visible neurons might be outdated
	///and must be computed again. 
	template<class BetaVector>
	void updateVisible(VisibleSampleBatch& visibleBatch, BetaVector const& newBeta)const{
		if(m_flags & StoreVisibleStatistics){
			mpe_rbm->visibleNeurons().sufficientStatistics(visibleBatch.input,visibleBatch.statistics, newBeta);
		}
	}


	///\brief Updates the value of the sufficient statistics of the sample after a change of temperature.
	///
	///After changing the temperature, requested information regarding the hidden neurons might be outdated
	///and must be computed again. 
	template<class BetaVector>
	void updateHidden(HiddenSampleBatch& hiddenBatch, BetaVector const& newBeta)const{
		if(m_flags & StoreHiddenStatistics){
			mpe_rbm->visibleNeurons().sufficientStatistics(hiddenBatch.input,hiddenBatch.statistics, newBeta);
		}
	}


	///\brief Creates a hidden/visible sample pair from a state of the visible neurons. this can directly be used to calculate the gradient.
	///
	/// @param hidden the hidden sample to be created
	/// @param visible the visible sample to be created
	/// @param state the state of the visible neurons in the sample
	/// @param beta the inverse temperature
	template<class States, class BetaVector>
	void createSample(HiddenSampleBatch& hiddenBatch,VisibleSampleBatch& visibleBatch, States const& states, BetaVector const& beta)const{
		SIZE_CHECK(size(states)==visibleBatch.size());
		SIZE_CHECK(hiddenBatch.size()==visibleBatch.size());
		visibleBatch.state = states;
		
		precomputeHidden(hiddenBatch,visibleBatch, beta);
		sampleHidden(hiddenBatch);
		//if one of the store visible flags (except state) is set, we also need to calculate the statistics of the visible given the hidden
		//this is not 100% correct, but better than nothing.
		// \todo @Oswin warum visible statistics und visible input speichern, wenn die fuers momentane sample sowieso nciht korrekt sind?
		if((m_flags & StoreHiddenFeatures) || (m_flags & StoreVisibleStatistics) || (m_flags & StoreVisibleInput)){
			precomputeVisible(hiddenBatch,visibleBatch, beta);
		}
	}
	
	///\brief Creates a hidden/visible sample pair from a state of the visible neurons. this can directly be used to calculate the gradient.
	///
	///In this version, the temperature defaults to 1.
	/// @param hidden the hidden sample to be created
	/// @param visible the visible sample to be created
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
		Energy energy(&mpe_rbm->structure());
		return energy.energyFromVisibleInput(
			visibleBatch.input, 
			hiddenBatch.state, 
			visibleBatch.state
		);
	}


private:
	RBM* mpe_rbm;
	SamplingFlags m_flags;
	
	///\brief Calculates the input of the hidden units and stores the visible features if requested
	void calculateHiddenInput(HiddenSampleBatch& hidden,VisibleSampleBatch& visible)const{
		Energy energy(&mpe_rbm->structure());
		if(m_flags & StoreVisibleFeatures){
			energy.inputHidden(hidden.input, visible.state, visible.features);
		}
		else{
			energy.inputHidden(hidden.input, visible.state);
		}
	}
	///\brief Calculates the input of the visible units and stores the hidden features if requested
	void calculateVisibleInput(HiddenSampleBatch& hidden, VisibleSampleBatch& visible)const{
		Energy energy(&mpe_rbm->structure());
		if(m_flags & StoreHiddenFeatures){
			energy.inputVisible(visible.input, hidden.state, hidden.features);
		}
		else{
			energy.inputVisible(visible.input, hidden.state);
		}
	}
};

	
}
#endif
