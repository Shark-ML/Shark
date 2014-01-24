/*!
 * 
 * \file        RNNet.cpp
 *
 * \brief       Recurrent Neural Network
 * 
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
#include <shark/Models/RNNet.h>

using namespace std;
using namespace shark;

void RNNet::eval(BatchInputType const& patterns, BatchOutputType& outputs, State& state)const{
	//initialize the history for the whole batch of sequences
	InternalState& s = state.toState<InternalState>();
	std::size_t warmUpLength=m_warmUpSequence.size();
	std::size_t numUnits = mpe_structure->numberOfUnits();
	s.timeActivation.resize(size(patterns));
	outputs.resize(size(patterns));

	//calculation of the sequences
	for(std::size_t b = 0; b != size(patterns);++b){
		std::size_t sequenceLength=size(get(patterns,b))+warmUpLength+1;
		s.timeActivation[b].resize(sequenceLength,RealVector(numUnits));
		outputs[b].resize(size(get(patterns,b)),RealVector(numUnits));
		Sequence& sequence = s.timeActivation[b];
		sequence[0].clear();
		for (std::size_t t = 1; t < sequenceLength;t++){
			//we want to treat input neurons exactly as hidden or output neurons, so we copy the current
			//pattern at the beginning of the the last activation pattern. After that, all activations
			//required for this timestep are in s.timeActivation[t-1]
			if(t<=warmUpLength)
				//we are still in warm up phase
				noalias(subrange(sequence[t-1],0,inputSize())) = m_warmUpSequence[t-1];
			else
				noalias(subrange(sequence[t-1],0,inputSize())) = patterns[b][t-1-warmUpLength];
			//and set the bias to 1
			sequence[t-1](mpe_structure->bias())=1;

			//activation of the hidden neurons is now just a matrix vector multiplication
			axpy_prod(
				mpe_structure->weights(),
				sequence[t-1],
				subrange(sequence[t],inputSize()+1,numUnits)
			);
			//now apply the sigmoid function
			for (std::size_t i = inputSize()+1;i != numUnits;i++)
				sequence[t](i) = mpe_structure->neuron(sequence[t](i));
			
			//if the warmup is over, we can copy the results into the output
			if(t>warmUpLength)
				outputs[b][t-1-warmUpLength] = subrange(sequence[t],numUnits-outputSize(),numUnits);
			
		}
	}
}

void RNNet::weightedParameterDerivative(
	BatchInputType const& patterns, BatchInputType const& coefficients, 
	State const& state, RealVector& gradient
)const{
	//SIZE_CHECK(pattern.size() == coefficients.size());
	InternalState const& s = state.toState<InternalState>();
	gradient.resize(numberOfParameters());
	gradient.clear();
	
	std::size_t numUnits = mpe_structure->numberOfUnits();
	std::size_t numNeurons = mpe_structure->numberOfNeurons();
	std::size_t warmUpLength=m_warmUpSequence.size();
	for(std::size_t b = 0; b != size(patterns); ++b){
		Sequence const& sequence = s.timeActivation[b];
		std::size_t sequenceLength = size(s.timeActivation[b]);
		RealMatrix errorDerivative(sequenceLength,numNeurons);
		errorDerivative.clear();
		//copy errors
		for (std::size_t t = warmUpLength+1; t != sequenceLength; ++t)
			for(std::size_t i = 0; i != outputSize(); ++i)
				errorDerivative(t,i+numNeurons-outputSize())=coefficients[b][t-warmUpLength-1](i);
		
		//backprop through time
		for (std::size_t t = (int)sequence.size()-1; t > 0; t--){
			for (std::size_t j = 0; j != numNeurons; ++j){
				double derivative = mpe_structure->neuronDerivative(sequence[t](j+mpe_structure->inputs()+1));
				errorDerivative(t,j)*=derivative;
			}
			axpy_prod(
				trans(columns(mpe_structure->weights(), inputSize()+1,numUnits)),
				row(errorDerivative,t),
				row(errorDerivative,t-1),
				false
			);
		}
		
		
		//update gradient for batch element i
		std::size_t param = 0;
		for (std::size_t i = 0; i != numNeurons; ++i){
			for (std::size_t j = 0; j != numUnits; ++j){
				if(!mpe_structure->connection(i,j))continue;

				for(std::size_t t=1;t != sequence.size(); ++t)
					gradient(param)+=errorDerivative(t,i) * sequence[t-1](j);
				
				++param;
			}
		}
		//sanity check
		SIZE_CHECK(param == mpe_structure->parameters());
	}
}

