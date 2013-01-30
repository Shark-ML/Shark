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
#ifndef SHARK_UNSUPERVISED_ENERGIES_ENERGYINTERACTIONTERM_H
#define SHARK_UNSUPERVISED_ENERGIES_ENERGYINTERACTIONTERM_H

#include "RBMStructure.h"
#include "AverageEnergyGradient1x1.h"
#include <shark/LinAlg/Base.h>

namespace shark{
namespace detail{

///\brief General definition for a interaction term of the energy function. 
///Must be specialized for different numbers of terms
template<class HiddenNeuron,class VisibleNeuron, class VectorType, std::size_t hiddenTerms, std::size_t visibleTerms>
class EnergyInteractionTerm{};

///\brief  The interaction term of the energy (the term of the energy function which tepends on the visible AND the hidden neurons).
///
///Uses the informations given by the neurons to automatize the calculation of energy and the derivative
///as well as for the factorization of the propability.
///This is the version for a simple interaction term of the form \f$ \phi_{h}(\vec h) W \phi_{v}(\vec v) \f$.
template<class HiddenNeuron,class VisibleNeuron,class VectorType>
struct EnergyInteractionTerm<HiddenNeuron,VisibleNeuron,VectorType, 1,1>{
public:
	typedef VectorType VisibleInput;
	typedef VectorType HiddenInput;
	typedef VectorType VisibleFeatures; 
	typedef VectorType HiddenFeatures;
	typedef typename VisibleNeuron::State VisibleState;
	typedef typename HiddenNeuron::State HiddenState;
	
	typedef typename Batch<VisibleInput>::type VisibleInputBatch;
	typedef typename Batch<HiddenInput>::type HiddenInputBatch;
	typedef typename Batch<VisibleFeatures>::type VisibleFeaturesBatch; 
	typedef typename Batch<HiddenFeatures>::type HiddenFeaturesBatch;
	typedef typename Batch<VisibleState>::type VisibleStateBatch; 
	typedef typename Batch<HiddenState>::type HiddenStateBatch;
	

	typedef RBMStructure< HiddenNeuron, VisibleNeuron, VectorType, 1, 1> Structure;
	typedef AverageEnergyGradient1x1<Structure> AverageEnergyGradient;

	/// Constructor
	EnergyInteractionTerm(Structure const* structure):mpe_structure(structure){}
		

	///\brief Calculates the inputs of the hidden neurons given the state of the visible.
	///
	///@param inputs The matix the inputs of the hidden neurons is stored in
	///@param visibleStates The Matrix of the states of the visible neurons
	void inputHidden(HiddenInputBatch& inputs, VisibleStateBatch const& visibleStates)const{
		SIZE_CHECK(size(visibleStates) == inputs.size1());
		SIZE_CHECK(inputs.size2() == mpe_structure->hiddenNeurons().size());
		
		//if calculating the value of the phi-function for the hidden units is expensive, 
		//it is worth storing the intermediate result before calculating the fast_prod
		if(VisibleNeuron::expensiveEvaluationOfPhi){
			std::size_t batchSize = inputs.size1();
			std::size_t visibleNeurons = mpe_structure->visibleNeurons().size();
			HiddenFeaturesBatch phiOfV(batchSize,visibleNeurons);
			inputHidden(inputs,visibleStates,phiOfV);
		}
		else{
			//since the elements form the row of the matrix, we have to transpose it
			fast_prod(mpe_structure->visibleNeurons().phi(visibleStates),trans(mpe_structure->weightMatrix(0,0)),inputs);
		}
	}


	///\brief Calculates the inputs of the visible neurons given the state of the hidden.
	///
	///@param inputs the batch the inputs of the visible neurons are stored in
	///@param hiddenStates the batch of the states of the hidden neurons
	void inputVisible(VisibleInputBatch& inputs, HiddenStateBatch const& hiddenStates)const{
		SIZE_CHECK(size(hiddenStates) == inputs.size1());
		SIZE_CHECK(inputs.size2() == mpe_structure->visibleNeurons().size());
		
		//if calculating the value of the phi-function of the visible unist is expensive,
		// it is worth storing the intermediate result before calculating the axpy_prod
		if(VisibleNeuron::expensiveEvaluationOfPhi){
			std::size_t batchSize = inputs.size1();
			std::size_t numHidden = mpe_structure->hiddenNeurons().size();
			VisibleFeaturesBatch phiOfH(batchSize,numHidden);
			inputVisible(inputs,hiddenStates,phiOfH);
		}
		else{
			fast_prod(mpe_structure->hiddenNeurons().phi(hiddenStates),mpe_structure->weightMatrix(0,0),inputs);
		}
	}
	

	///\brief Calculates the inputs of the hidden neurons given the state of the visible.
	/// and stores the value of the phi-function of the visible neurons
	///
	///@param inputs the matrix the inputs of the hidden neurons is stored in
	///@param visibleStates the matrix of states of the visible neurons
	///@param phiOfV the matrix the value of the phi-function given the actual state of the visible neurons is stored in
	void inputHidden(HiddenInputBatch& inputs, VisibleStateBatch const& visibleStates, VisibleFeaturesBatch& phiOfV)const{
		SIZE_CHECK(size(inputs) == size(visibleStates));
		SIZE_CHECK(size(inputs) == size(phiOfV));
		SIZE_CHECK(inputs.size2() == mpe_structure->hiddenNeurons().size());
		SIZE_CHECK(phiOfV.size2() == mpe_structure->visibleNeurons().size());
		
		noalias(phiOfV) = mpe_structure->visibleNeurons().phi(visibleStates);
		fast_prod(phiOfV,trans(mpe_structure->weightMatrix(0,0)),inputs);
	}


	///\brief Calculates the inputs of the visible neurons given the state of the hidden.
	///and stores the value of the phi-function of the hidden neurons
	///
	///@param inputs the batch the inputs of the visible neurons are stored in
	///@param hiddenStates the batch of states of the hidden neurons
	///@param phiOfH the batch the values of the phi-function given the actual state of the visible neurons is stored in
	void inputVisible(VisibleInputBatch& inputs, HiddenStateBatch const& hiddenStates, HiddenFeaturesBatch& phiOfH)const{
		SIZE_CHECK(size(inputs) == size(hiddenStates));
		SIZE_CHECK(size(inputs) == size(phiOfH));
		SIZE_CHECK(inputs.size2() == mpe_structure->visibleNeurons().size());
		SIZE_CHECK(phiOfH.size2() == mpe_structure->hiddenNeurons().size());
		
		noalias(phiOfH) =  mpe_structure->hiddenNeurons().phi(hiddenStates);
		fast_prod(phiOfH,mpe_structure->weightMatrix(0,0),inputs);
	}
	

	///\brief Optimization of the calculation of the energy, when the inputs of the hidden units is.
	/// and the value of the phi-function of the hidden neurons is already available.
	///
	///@param hiddenInput the inputs of the hidden neurons
	///@param hidden the state of the hidden neurons
 	///@param visible the state of the visible neurons
	///@param phiOfH the values of the phi-function of the hidden neurons
	///@return the value of the energy function
	template<class Input,class StateHidden,class StateVisible,class Features>
	VectorType energyFromHiddenInput(
		Input const& hiddenInput,
		StateHidden const& hidden, 
		StateVisible const& visible, 
		Features const& phiOfH
	)const{
		std::size_t batchSize = size(hiddenInput);
		VectorType energies(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			energies(i) = -inner_prod(row(hiddenInput,i),row(phiOfH,i));
		}
		energies -= mpe_structure->hiddenNeurons().energyTerm(hidden);
		energies -= mpe_structure->visibleNeurons().energyTerm(visible);
		return energies;
	}


	///\brief Optimization of the calculation of the energy, when the inputs of the visible units.
    /// and the value of the phi-function of the visible neurons is already available.
	///
	///@param visibleInput the inputs of the visible neurons
	///@param hidden the state of the hidden neurons
 	///@param visible the state of the visible neurons
	///@param phiOfV the values of the phi-function of the visible neurons
	///@return the value of the energy function
	template<class Input,class StateHidden,class StateVisible,class Features>
	VectorType energyFromVisibleInput(
		Input const& visibleInput, 
		StateHidden const& hidden, 
		StateVisible const& visible,
		Features const& phiOfV
	)const{
		std::size_t batchSize = size(visibleInput);
		VectorType energies(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			energies(i) = -inner_prod(row(phiOfV,i),row(visibleInput,i));
		}
		energies -= mpe_structure->hiddenNeurons().energyTerm(hidden);
		energies -= mpe_structure->visibleNeurons().energyTerm(visible);
		return energies;
	}


	///\brief Optimization of the calculation of the energy, when the inputs of the hidden units is already available.
	///@param hiddenInput the vector of inputs of the hidden neurons
	///@param hidden the state of the hidden neurons
 	///@param visible the state of the visible neurons
	///@return the value of the energy function
	template<class Input,class Vector1,class Vector2>
	VectorType energyFromHiddenInput(
		Input const& hiddenInput,
		Vector1 const& hidden, 
		Vector2 const& visible
	)const{
		return energyFromHiddenInput(hiddenInput,hidden,visible, mpe_structure->hiddenNeurons().phi(hidden));
	}


	///\brief Optimization of the calculation of the energy, when the inputs of the visible units is already available.
	///@param visibleInput the vector of inputs of the visible neurons
	///@param hidden the states of the hidden neurons
 	///@param visible the states of the visible neurons
	///@return the value of the energy function
	template<class Input,class Vector1,class Vector2>
	VectorType energyFromVisibleInput(
		Input const& visibleInput, 
		Vector1 const& hidden, 
		Vector2 const& visible
	)const{
 		return energyFromVisibleInput(visibleInput,hidden,visible, mpe_structure->visibleNeurons().phi(visible));
	}

protected:
	const Structure* mpe_structure;
};
	
}	
}

#endif
