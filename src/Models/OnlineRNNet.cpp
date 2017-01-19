/*!
 * 
 *
 * \brief       Recurrent Neural Network
 * 
 * \par Copyright (c) 1999-2007:
 * Institut f&uuml;r Neuroinformatik
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
#define SHARK_COMPILE_DLL

#include <shark/Models/OnlineRNNet.h>
using namespace std;
using namespace shark;

OnlineRNNet::OnlineRNNet(RecurrentStructure* structure, bool computeGradient)
:mpe_structure(structure), m_computeGradient(computeGradient){
	SHARK_CHECK(mpe_structure,"[OnlineRNNet] structure pointer is not allowed to be NULL");
	if(computeGradient)
		m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
}


void OnlineRNNet::eval(RealMatrix const& pattern, RealMatrix& output, State& state)const{
	SIZE_CHECK(pattern.size1()==1);//we can only process a single input at a time.
	SIZE_CHECK(pattern.size2() == inputSize());
	
	
	std::size_t numNeurons = mpe_structure->numberOfNeurons();
	std::size_t numUnits = mpe_structure->numberOfUnits();
	InternalState& s = state.toState<InternalState>();
	RealVector& lastActivation = s.lastActivation;
	RealVector& activation = s.activation;
	swap(lastActivation,activation);

	//we want to treat input and bias neurons exactly as hidden or output neurons, so we copy the current
	//pattern at the beginning of the the last activation pattern aand set the bias neuron to 1
	////so lastActivation has the format (input|1|lastNeuronActivation)
	noalias(subrange(lastActivation,0,mpe_structure->inputs())) = row(pattern,0);
	lastActivation(mpe_structure->bias())=1;
	activation(mpe_structure->bias())=1;

	//activation of the hidden neurons is now just a matrix vector multiplication

	noalias(subrange(activation,inputSize()+1,numUnits)) = prod(
		mpe_structure->weights(),
		lastActivation
	);

	//now apply the sigmoid function
	for (std::size_t i = inputSize()+1;i != numUnits;i++){
		activation(i) = mpe_structure->neuron(activation(i));
	}
	//copy the result to the output
	output.resize(1,outputSize());
	noalias(row(output,0)) = subrange(activation,numUnits-outputSize(),numUnits);
	
	//update the internal derivative if needed
	if(!m_computeGradient) return;
	
	RealMatrix& unitGradient = s.unitGradient;
	
	//for the next steps see Kenji Doya, "Recurrent Networks: Learning Algorithms"

	//calculate the derivative for all neurons f'
	RealVector neuronDerivatives(numNeurons);
	for(std::size_t i=0;i!=numNeurons;++i){
		neuronDerivatives(i)=mpe_structure->neuronDerivative(activation(i+inputSize()+1));
	}
	
	//calculate the derivative for every weight using the derivative of the last time step
	auto hiddenWeights = columns(
		mpe_structure->weights(),
		inputSize()+1,numUnits
	);
	
	//update the new gradient with the effect of last timestep
	unitGradient = prod(unitGradient,trans(hiddenWeights));
	
	//add the effect of the current time step when there is a connection
	std::size_t param = 0;
	for(std::size_t i = 0; i != numNeurons; ++i){
		for(std::size_t j = 0; j != numUnits; ++j){
			if(mpe_structure->connection(i,j)){
				unitGradient(param,i) += lastActivation(j);
			}
		}
	}
	
	//multiply with outer derivative of the neurons
	for(std::size_t i = 0; i != unitGradient.size1();++i){
		noalias(row(unitGradient,i)) *= neuronDerivatives;
	}
	
	//We are done here for eval, the rest can only be computed using an error signal
}


void OnlineRNNet::weightedParameterDerivative(RealMatrix const& pattern, const RealMatrix& coefficients,  State const& state, RealVector& gradient)const{
	if(!m_computeGradient) throw SHARKEXCEPTION("[OnlineFFNet::weightedParameterDerivative] Network is configured to not computing gradients!");
	SIZE_CHECK(pattern.size1()==1);//we can only process a single input at a time.
	SIZE_CHECK(coefficients.size1()==1);
	SIZE_CHECK(pattern.size2() == inputSize());
	SIZE_CHECK(pattern.size2() == coefficients.size2());
	gradient.resize(mpe_structure->parameters());
	
	std::size_t numNeurons = mpe_structure->numberOfNeurons();
	InternalState const& s = state.toState<InternalState>();
	RealMatrix const& unitGradient = s.unitGradient;
	//and formula 4 (the gradient itself)
	noalias(gradient) = prod(
		columns(unitGradient,numNeurons-outputSize(),numNeurons),
		row(coefficients,0)
	);
}
