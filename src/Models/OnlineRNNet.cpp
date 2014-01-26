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


#include <shark/Models/OnlineRNNet.h>
using namespace std;
using namespace shark;

OnlineRNNet::OnlineRNNet(RecurrentStructure* structure):mpe_structure(structure),m_unitGradient(0,0){
	SHARK_CHECK(mpe_structure,"[OnlineRNNet] structure pointer is not allowed to be NULL");
	m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
}


void OnlineRNNet::eval(RealMatrix const& pattern, RealMatrix& output){
	SIZE_CHECK(pattern.size1()==1);//we can only process a single input at a time.
	SIZE_CHECK(pattern.size2() == inputSize());
	
	std::size_t numUnits = mpe_structure->numberOfUnits();
	
	if(m_lastActivation.size() != numUnits){
		m_activation.resize(numUnits);
		m_lastActivation.resize(numUnits);
		m_activation.clear();
		m_lastActivation.clear();
	}
	swap(m_lastActivation,m_activation);

	//we want to treat input and bias neurons exactly as hidden or output neurons, so we copy the current
	//pattern at the beginning of the the last activation pattern aand set the bias neuron to 1
	////so m_lastActivation has the format (input|1|lastNeuronActivation)
	noalias(subrange(m_lastActivation,0,mpe_structure->inputs())) = row(pattern,0);
	m_lastActivation(mpe_structure->bias())=1;
	m_activation(mpe_structure->bias())=1;

	//activation of the hidden neurons is now just a matrix vector multiplication

	axpy_prod(
		mpe_structure->weights(),
		m_lastActivation,
		subrange(m_activation,inputSize()+1,numUnits)
	);

	//now apply the sigmoid function
	for (std::size_t i = inputSize()+1;i != numUnits;i++){
		m_activation(i) = mpe_structure->neuron(m_activation(i));
	}
	//copy the result to the output
	output.resize(1,outputSize());
	noalias(row(output,0)) = subrange(m_activation,numUnits-outputSize(),numUnits);
}


void OnlineRNNet::weightedParameterDerivative(RealMatrix const& pattern, const RealMatrix& coefficients,  RealVector& gradient){
	SIZE_CHECK(pattern.size1()==1);//we can only process a single input at a time.
	SIZE_CHECK(coefficients.size1()==1);
	SIZE_CHECK(pattern.size2() == inputSize());
	SIZE_CHECK(pattern.size2() == coefficients.size2());
	gradient.resize(mpe_structure->parameters());
	
	std::size_t numNeurons = mpe_structure->numberOfNeurons();
	std::size_t numUnits = mpe_structure->numberOfUnits();

	//first check wether this is the first call of the derivative after a change of internal structure. in this case we have to allocate A LOT
	//of memory for the derivative and set it to zero.
	if(m_unitGradient.size1() != mpe_structure->parameters() || m_unitGradient.size2() != numNeurons){
		m_unitGradient.resize(mpe_structure->parameters(),numNeurons);
		m_unitGradient.clear();
	}

	//for the next steps see Kenji Doya, "Recurrent Networks: Learning Algorithms"

	//calculate the derivative for all neurons f'
	RealVector neuronDerivatives(numNeurons);
	for(std::size_t i=0;i!=numNeurons;++i){
		neuronDerivatives(i)=mpe_structure->neuronDerivative(m_activation(i+inputSize()+1));
	}
	
	//calculate the derivative for every weight using the derivative of the last time step
	ConstRealSubMatrix hiddenWeights = columns(
		mpe_structure->weights(),
		inputSize()+1,numUnits
	);
	
	//update the new gradient with the effect of last timestep
	RealMatrix newUnitGradient(mpe_structure->parameters(),numNeurons);
	axpy_prod(m_unitGradient,trans(hiddenWeights),newUnitGradient);
	swap(m_unitGradient,newUnitGradient);
	newUnitGradient = RealMatrix();//empty
	
	
	//add the effect of the current time step
	std::size_t param = 0;
	for(std::size_t i = 0; i != numNeurons; ++i){
		for(std::size_t j = 0; j != numUnits; ++j){
			if(mpe_structure->connection(i,j)){
				m_unitGradient(param,i)+=m_lastActivation(j);
				++param;
			}
		}
	}
	
	//multiply with outer derivative of the neurons
	for(std::size_t i = 0; i != m_unitGradient.size1();++i){
		noalias(row(m_unitGradient,i))= element_prod(row(m_unitGradient,i),neuronDerivatives);
	}
	//and formula 4 (the gradient itself)
	axpy_prod(
		columns(m_unitGradient,numNeurons-outputSize(),numNeurons),
		row(coefficients,0),
		gradient
	);
	//sanity check
	SIZE_CHECK(param == mpe_structure->parameters());
}
