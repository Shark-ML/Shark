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
#ifndef SHARK_UNSUPERVISED_RBM_ENERGY_H
#define SHARK_UNSUPERVISED_RBM_ENERGY_H

#include "Impl/EnergyInteractionTerm.h"

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
	
template<class Visible, class Hidden, class VectorT = RealVector>
struct Energy{
private:
	typedef detail::EnergyInteractionTerm< Hidden,Visible,VectorT, Hidden::activationTerms,Visible::activationTerms> InteractionTerm;
public:
	typedef typename InteractionTerm::Structure Structure;
	typedef Hidden HiddenType; //< type of the hidden layer
	typedef Visible VisibleType; //< type of the visible layer
	
	typedef VectorT VectorType;
	
	//typedefs for single element
	typedef typename InteractionTerm::VisibleInput VisibleInput;
	typedef typename InteractionTerm::HiddenInput HiddenInput;
	typedef typename InteractionTerm::VisibleFeatures VisibleFeatures; 
	typedef typename InteractionTerm::HiddenFeatures HiddenFeatures;
	typedef typename InteractionTerm::VisibleState VisibleState; 
	typedef typename InteractionTerm::HiddenState HiddenState;
	typedef typename HiddenType::SufficientStatistics HiddenStatistics;
	typedef typename VisibleType::SufficientStatistics VisibleStatistics;
	
	//batch typedefs
	typedef typename InteractionTerm::VisibleInputBatch VisibleInputBatch;
	typedef typename InteractionTerm::HiddenInputBatch HiddenInputBatch;
	typedef typename InteractionTerm::VisibleFeaturesBatch VisibleFeaturesBatch; 
	typedef typename InteractionTerm::HiddenFeaturesBatch HiddenFeaturesBatch;
	typedef typename InteractionTerm::VisibleStateBatch VisibleStateBatch; 
	typedef typename InteractionTerm::HiddenStateBatch HiddenStateBatch;
	typedef typename Batch<HiddenStatistics>::type HiddenStatisticsBatch;
	typedef typename Batch<VisibleStatistics>::type VisibleStatisticsBatch;
	
	//gradient of the derivative
	typedef typename InteractionTerm::AverageEnergyGradient AverageEnergyGradient;
	

	Energy(const Structure* structure):m_interaction(structure), mpe_structure(structure){}
		
	///\brief Calculates the Energy given the states of batches of hidden and visible variables .
	VectorType energy(HiddenStateBatch const& hidden, VisibleStateBatch const& visible)const{
		SIZE_CHECK(size(visible)==size(hidden));
		
		std::size_t batchSize = size(visible);
		HiddenInputBatch input(batchSize,mpe_structure->numberOfHN());
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
		HiddenStateBatch const& hiddenState, 
		VisibleInputBatch const& visibleInput, 
		BetaVector beta
	)const{
		SIZE_CHECK(size(hiddenState)==size(visibleInput));
		SIZE_CHECK(size(hiddenState)==beta.size());
		std::size_t batchSize = size(hiddenState);
		
		//calculate the energy terms of the hidden neurons for the whole batch
		RealVector energyTerms = mpe_structure->hiddenNeurons().energyTerm(hiddenState);

		//calculate resulting probabilities in sequence
		RealVector p(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			double hiddenEnergy = beta(i) * energyTerms(i);
			p(i) = mpe_structure->visibleNeurons().logMarginalize(get(visibleInput,i),beta(i))+hiddenEnergy;
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
		VisibleStateBatch const& visibleState,
		HiddenInputBatch const& hiddenInput, 
		BetaVector const& beta
	)const{
		SIZE_CHECK(size(visibleState)==size(hiddenInput));
		SIZE_CHECK(size(visibleState)==beta.size());
		std::size_t batchSize = size(visibleState);
		
		//calculate the energy terms of the visible neurons for the whole batch
		RealVector energyTerms = mpe_structure->visibleNeurons().energyTerm(visibleState);
	
		
		RealVector p(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			double visibleEnergy = beta(i) * energyTerms(i);
			p(i) = mpe_structure->hiddenNeurons().logMarginalize(get(hiddenInput,i),beta(i))+visibleEnergy;
		}
		return p;
	}

	
	///\brief Computes the logarithm of the unnormalized probability for each state of the visible neurons from a batch.
	///
	///@param visibleState the batch of states of the hidden neurons
	///@param beta the inverse temperature
	template<class BetaVector>
	RealVector logUnnormalizedPropabilityVisible(VisibleStateBatch const& visibleStates, BetaVector const& beta)const{
		SIZE_CHECK(size(visibleStates)==beta.size());
		
		HiddenInputBatch hiddenInputs(beta.size(),mpe_structure->numberOfHN());
		inputHidden(hiddenInputs,visibleStates);
		return logUnnormalizedPropabilityVisible(visibleStates, hiddenInputs, beta);
	}
	
	///\brief Computes the logarithm of the unnormalized probability of each state of the hidden neurons from a batch.
	///
	///@param hiddenStates a bacth of states of the hidden neurons
	///@param beta the inverse temperature
	template<class BetaVector>
	RealVector logUnnormalizedPropabilityHidden(HiddenStateBatch const& hiddenStates, BetaVector const& beta)const{
		SIZE_CHECK(size(hiddenStates)==beta.size());
		
		VisibleInputBatch visibleInputs(beta.size(),mpe_structure->numberOfVN());
		inputVisible(visibleInputs,hiddenStates);
		return logUnnormalizedPropabilityHidden(hiddenStates, visibleInputs, beta);
	}
    
    
	///\brief Calculates the input of the hidden neurons given the state of the visible in a batch-vise fassion.
	///
	///@param input the batch of vectors the input of the hidden neurons is stored in
	///@param visibleState the batch of states of the visible neurons
	void inputHidden(HiddenInputBatch& input, VisibleStateBatch const& visibleState)const{
		m_interaction.inputHidden(input, visibleState);
	}


	///\brief Calculates the input of the visible neurons given the state of the hidden.
	///
	///@param input the vector the input of the visible neurons is stored in
	///@param hiddenState the state of the hidden neurons
	void inputVisible(VisibleInputBatch& input, HiddenStateBatch const& hiddenState)const{
		m_interaction.inputVisible(input,hiddenState);
	}
	

	///\brief Calculates the input of the hidden neurons given the state of the visible.
	/// This version also stores the value of the phi-function of the visible neurons
	///
	///@param input the vector the input of the hidden neurons is stored in
	///@param visibleState the state of the visible neurons
	///@param phiOfV the vector the value of the phi-function given the actual state of the visible neurons is stored in
	void inputHidden(HiddenInputBatch& input, VisibleStateBatch const& visibleState, VisibleFeaturesBatch& phiOfV)const{
		m_interaction.inputHidden(input, visibleState,phiOfV);
	}


	///\brief Calculates the input of the visible neurons given the state of the hidden.
	///and stores the value of the phi-function of the hidden neurons
	///
	///@param input the vector the input of the visible neurons is stored in
	///@param hiddenState the state of the hidden neurons
	///@param phiOfH the vector the value of the phi-function given the actual state of the visible neurons is stored in
	void inputVisible(VisibleInputBatch& input, HiddenStateBatch const& hiddenStates, HiddenFeaturesBatch& phiOfH)const{
		m_interaction.inputVisible(input,hiddenStates, phiOfH);
	}
	

	///\brief Optimization of the calculation of the energy, when the input of the hidden units
	/// and the value of the phi-function of the hidden neurons is already available.
	///
	///@param hiddenInput the vector of inputs of the hidden neurons
	///@param hidden the states of the hidden neurons
 	///@param visible the states of the visible neurons
	///@param phiOfH the values of the phi-function of the hidden neurons
	///@return the value of the energy function
	VectorType energyFromHiddenInput(
		HiddenInputBatch const& hiddenInput,
		HiddenStateBatch const& hidden, 
		VisibleStateBatch const& visible,
		HiddenFeaturesBatch const& phiOfH
	)const{
		return m_interaction.energyFromHiddenInput(hiddenInput,hidden,visible,phiOfH);
	}


	///\brief Optimization of the calculation of the energy, when the input of the visible units.
    /// and the value of the phi-function of the visible neurons is already available.
	///
	///@param visibleInput the vector of inputs of the visible neurons
	///@param hidden the states of the hidden neurons
 	///@param visible the states of the visible neurons
	///@param phiOfV the values of the phi-function of the visible neurons
	///@return the value of the energy function
	VectorType energyFromVisibleInput(
		VisibleInputBatch const& visibleInput,
		HiddenStateBatch const& hidden, 
		VisibleStateBatch const& visible,
		VisibleFeaturesBatch const& phiOfV
	)const{
		return m_interaction.energyFromVisibleInput(visibleInput,hidden,visible,phiOfV);
	}


	///\brief Optimization of the calculation of the energy, when the input of the hidden units is already available.
	///@param hiddenInput the vector of inputs of the hidden neurons
	///@param hidden the states of the hidden neurons
 	///@param visible the states of the visible neurons
	///@return the value of the energy function
	VectorType energyFromHiddenInput(
		HiddenInputBatch const& hiddenInput,
		HiddenStateBatch const& hidden, 
		VisibleStateBatch const& visible
	)const{
		return m_interaction.energyFromHiddenInput(hiddenInput,hidden,visible);
	}


	///\brief Optimization of the calculation of the energy, when the input of the visible units is already available.
	///@param visibleInput the vector of inputs of the visible neurons
	///@param hidden the states of the hidden neurons
 	///@param visible the states of the visible neurons
	///@return the value of the energy function
	VectorType energyFromVisibleInput(
		VisibleInputBatch const& visibleInput,
		HiddenStateBatch const& hidden, 
		VisibleStateBatch const& visible
	)const{
 		return m_interaction.energyFromVisibleInput(visibleInput,hidden,visible);
	}
private:
	InteractionTerm m_interaction;
	Structure const* mpe_structure;
	
};

}

#endif
