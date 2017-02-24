/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_UNSUPERVISED_RBm_ENERGY_H
#define SHARK_UNSUPERVISED_RBm_ENERGY_H

#include <shark/LinAlg/Base.h>
#include <shark/Data/BatchInterface.h>

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

	Energy(RBM const& rbm)
	: m_rbm(rbm)
	, m_hiddenNeurons(rbm.hiddenNeurons())
	, m_visibleNeurons(rbm.visibleNeurons()){}
		
	///\brief Calculates the Energy given the states of batches of hidden and visible variables .
	RealVector energy(RealMatrix const& hidden, RealMatrix const& visible)const{
		SIZE_CHECK(visible.size1() == hidden.size1());
		
		std::size_t batchSize = visible.size1();
		RealMatrix input(batchSize,m_hiddenNeurons.size());
		inputHidden( input, visible);
		
		return energyFromHiddenInput( input, hidden, visible);
	}
	
	///\brief Calculates the input of the hidden neurons given the state of the visible in a batch-vise fassion.
	///
	///@param inputs the batch of vectors the input of the hidden neurons is stored in
	///@param visibleStates the batch of states of the visible neurons@
	///@todo Remove this and replace fully by the rbm method if possible
	void inputHidden(RealMatrix& inputs, RealMatrix const& visibleStates)const{
		m_rbm.inputHidden(inputs,visibleStates);
	}


	///\brief Calculates the input of the visible neurons given the state of the hidden.
	///
	///@param inputs the vector the input of the visible neurons is stored in
	///@param hiddenStates the state of the hidden neurons
	///@todo Remove this and replace fully by the rbm method if possible
	void inputVisible(RealMatrix& inputs, RealMatrix const& hiddenStates)const{
		m_rbm.inputVisible(inputs,hiddenStates);
	}
	
	///\brief Computes the logarithm of the unnormalized probability of each state of the
	/// hidden neurons in a batch by using the precomputed input/activation of the visible neurons.
	///
	///@param hiddenState the batch of states of the hidden neurons
	///@param visibleInput the batch of current inputs for he visible units given hiddenState
	///@param beta the inverse temperature
	///@return the unnormalized probability
	template<class BetaVector>
	RealVector logUnnormalizedProbabilityHidden(
		RealMatrix const& hiddenState, 
		RealMatrix const& visibleInput, 
		BetaVector const& beta
	)const{
		SIZE_CHECK(hiddenState.size1()==visibleInput.size1());
		SIZE_CHECK(hiddenState.size1()==beta.size());
		std::size_t batchSize = hiddenState.size1();
		
		//calculate the energy terms of the hidden neurons for the whole batch
		RealVector energyTerms = m_hiddenNeurons.energyTerm(hiddenState,beta);

		//calculate resulting probabilities in sequence
		RealVector p(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			p(i) = m_visibleNeurons.logMarginalize(row(visibleInput,i),beta(i))+energyTerms(i);
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
	RealVector logUnnormalizedProbabilityVisible(
		RealMatrix const& visibleState,
		RealMatrix const& hiddenInput, 
		BetaVector const& beta
	)const{
		SIZE_CHECK(visibleState.size1()==hiddenInput.size1());
		SIZE_CHECK(visibleState.size1()==beta.size());
		std::size_t batchSize = visibleState.size1();
		
		//calculate the energy terms of the visible neurons for the whole batch
		RealVector energyTerms = m_visibleNeurons.energyTerm(visibleState,beta);
		
		RealVector p(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			p(i) = m_hiddenNeurons.logMarginalize(row(hiddenInput,i),beta(i))+energyTerms(i);
		}
		return p;
	}

	
	///\brief Computes the logarithm of the unnormalized probability for each state of the visible neurons from a batch.
	///
	///@param visibleStates the batch of states of the hidden neurons
	///@param beta the inverse temperature
	template<class BetaVector>
	RealVector logUnnormalizedProbabilityVisible(RealMatrix const& visibleStates, BetaVector const& beta)const{
		SIZE_CHECK(visibleStates.size1() == beta.size());
		
		RealMatrix hiddenInputs(beta.size(),m_hiddenNeurons.size());
		inputHidden(hiddenInputs,visibleStates);
		return logUnnormalizedProbabilityVisible(visibleStates, hiddenInputs, beta);
	}
	
	///\brief Computes the logarithm of the unnormalized probability of each state of the hidden neurons from a batch.
	///
	///@param hiddenStates a batch of states of the hidden neurons
	///@param beta the inverse temperature
	template<class BetaVector>
	RealVector logUnnormalizedProbabilityHidden(RealMatrix const& hiddenStates, BetaVector const& beta)const{
		SIZE_CHECK(hiddenStates.size1() == beta.size());
		
		RealMatrix visibleInputs(beta.size(),m_visibleNeurons.size());
		inputVisible(visibleInputs,hiddenStates);
		return logUnnormalizedProbabilityHidden(hiddenStates, visibleInputs, beta);
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
		RealMatrix const& phiOfH = m_hiddenNeurons.phi(hidden);
		std::size_t batchSize = hiddenInput.size1();
		RealVector energies(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			energies(i) = -inner_prod(row(hiddenInput,i),row(phiOfH,i));
		}
		energies -= m_hiddenNeurons.energyTerm(hidden,blas::repeat(1.0,batchSize));
		energies -= m_visibleNeurons.energyTerm(visible,blas::repeat(1.0,batchSize));
		return energies;
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
 		RealMatrix const& phiOfV = m_visibleNeurons.phi(visible);
		std::size_t batchSize = visibleInput.size1();
		RealVector energies(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			energies(i) = -inner_prod(row(phiOfV,i),row(visibleInput,i));
		}
		energies -= m_hiddenNeurons.energyTerm(hidden,blas::repeat(1.0,batchSize));
		energies -= m_visibleNeurons.energyTerm(visible,blas::repeat(1.0,batchSize));
		return energies;
	}
private:
	RBM const& m_rbm;
	HiddenType const& m_hiddenNeurons;
	VisibleType const& m_visibleNeurons;	
};

}

#endif
