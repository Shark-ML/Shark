/*!
 * \brief   Implements the autoencoder with tied weights
 * 
 * \author      O. Krause
 * \date        2010-2014
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
#ifndef SHARK_MODELS_TIEDAUTOENCODER_H
#define SHARK_MODELS_TIEDAUTOENCODER_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/Neurons.h>
#include <boost/serialization/vector.hpp>

namespace shark{

/// \brief implements the autoencoder with tied weights
///
/// The formula is
///  \f[ f(x) = \sigma_2(W^T\sigma_1(Wx+b_1)+b_2)\f]
/// Where \f$ W \f$, \f$b_1 \f$ and \f$b_2 \f$ are the weights and
///  \f$\sigma_1\f$ and \f$ \sigma_2\f$ are the activation functions for hidden and output units.
template<class HiddenNeuron,class OutputNeuron>
class TiedAutoencoder :public AbstractModel<RealVector,RealVector>
{
	struct InternalState: public State{
		RealMatrix hiddenResponses;
		RealMatrix outputResponses;
	};
	

public:
	TiedAutoencoder(){
		m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
		m_features|=HAS_FIRST_INPUT_DERIVATIVE;
	}

	//! \brief From INameable: return the class name.
	std::string name() const{ 
		return "TiedAutoencoder";
	}

	//! \brief Number of input neurons.
	std::size_t inputSize()const{
		return outputSize();
	}
	//! \brief Number of output neurons.
	std::size_t outputSize()const{
		return outputBias().size();
	}
	
	//! \brief Total number of hidden neurons.
	std::size_t numberOfHiddenNeurons()const{
		return encoderMatrix().size1();
	}
	
	/// \brief Returns the hidden bias weight vector.
	RealVector const& hiddenBias()const{
		return m_hiddenBias;
	}
	
	/// \brief Returns the hidden bias weight vector.
	RealVector& hiddenBias(){
		return m_hiddenBias;
	}
	
	/// \brief Returns the output bias weight vector.
	RealVector const& outputBias()const{
		return m_outputBias;
	}
	/// \brief Returns the output bias weight vector.
	RealVector& outputBias(){
		return m_outputBias;
	}
	
	/// \brief Weight matrix for the direction input->hidden.
	RealMatrix const& encoderMatrix()const{
		return m_weightMatrix;
	}
	/// \brief Weight matrix for the direction input->hidden.
	RealMatrix& encoderMatrix(){
		return m_weightMatrix;
	}
	/// \brief Weight matrix for the direction hidden->output
	///
	///For tied autoencoders, this is the transpose of the encoder matrix
	blas::matrix_transpose<RealMatrix const> decoderMatrix()const{
		return trans(m_weightMatrix);
	}
	/// \brief Weight matrix for the direction hidden->output
	///
	///For tied autoencoders, this is the transpose of the encoder matrix
	blas::matrix_transpose<RealMatrix> decoderMatrix(){
		return trans(m_weightMatrix);
	}
	
	//! \brief Returns the total number of parameters of the network. 
	std::size_t numberOfParameters()const{
		return inputSize()*numberOfHiddenNeurons()+inputSize()+numberOfHiddenNeurons();
	}

	//! returns the vector of used parameters inside the weight matrix
	RealVector parameterVector() const{
		RealVector parameters(numberOfParameters());
		init(parameters) << toVector(m_weightMatrix),m_hiddenBias,m_outputBias;
		return parameters;
	}
	//! uses the values inside the parametervector to set the used values inside the weight matrix
	void setParameterVector(RealVector const& newParameters){
		SIZE_CHECK(newParameters.size() == numberOfParameters());
		init(newParameters) >> toVector(m_weightMatrix),m_hiddenBias,m_outputBias;
	}
	
	/// \brief Returns the activation function of the hidden units.
	HiddenNeuron const& hiddenActivationFunction()const{
		return m_hiddenNeuron;
	}
	/// \brief Returns the activation function of the output units.
	OutputNeuron const& outputActivationFunction()const{
		return m_outputNeuron;
	}
	
	/// \brief Returns the activation function of the hidden units.
	HiddenNeuron& hiddenActivationFunction(){
		return m_hiddenNeuron;
	}
	/// \brief Returns the activation function of the output units.
	OutputNeuron& outputActivationFunction(){
		return m_outputNeuron;
	}

	//! \brief Returns the output of all neurons after the last call of eval
	//!
	//!     \param  state last result of eval
	//!     \return Output value of the neurons.
	RealMatrix const& hiddenResponses(State const& state)const{
		InternalState const& s = state.toState<InternalState>();
		return s.hiddenResponses;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}

	void evalLayer(std::size_t layer,RealMatrix const& patterns,RealMatrix& outputs)const{
		SIZE_CHECK(layer < 2);
		std::size_t numPatterns = patterns.size1();
		
		if(layer == 0){//input->hidden
			SIZE_CHECK(patterns.size2() == encoderMatrix().size2());
			std::size_t numOutputs = encoderMatrix().size1();
			outputs.resize(numPatterns,numOutputs);
			outputs.clear();
			axpy_prod(patterns,trans(encoderMatrix()),outputs);
			noalias(outputs) += repeat(hiddenBias(),numPatterns);
			noalias(outputs) = m_hiddenNeuron(outputs);
		}
		else{//hidden->output
			SIZE_CHECK(patterns.size2() == decoderMatrix().size2());
			std::size_t numOutputs = decoderMatrix().size1();
			outputs.resize(numPatterns,numOutputs);
			outputs.clear();
			axpy_prod(patterns,trans(decoderMatrix()),outputs);
			noalias(outputs) += repeat(outputBias(),numPatterns);
			noalias(outputs) = m_outputNeuron(outputs);
		}
	}
	
	///\brief Returns the response of the i-th layer given the input of that layer.
	///
	/// this is usefull if only a portion of the network needs to be evaluated
	/// be aware that this only works without shortcuts in the network
	Data<RealVector> evalLayer(std::size_t layer, Data<RealVector> const& patterns)const{
		SIZE_CHECK(layer < 2);
		int batches = (int) patterns.numberOfBatches();
		Data<RealVector> result(batches);
		SHARK_PARALLEL_FOR(int i = 0; i < batches; ++i){
			evalLayer(layer,patterns.batch(i),result.batch(i));
		}
		return result;
	}
	
	void eval(RealMatrix const& patterns,RealMatrix& output, State& state)const{
		InternalState& s = state.toState<InternalState>();
		evalLayer(0,patterns,s.hiddenResponses);//propagate input->hidden
		evalLayer(1,s.hiddenResponses,s.outputResponses);//propagate hidden->output
		output = s.outputResponses;
	}
	using AbstractModel<RealVector,RealVector>::eval;

	void weightedParameterDerivative(
		BatchInputType const& patterns, RealMatrix const& coefficients, State const& state, RealVector& gradient
	)const{
		SIZE_CHECK(coefficients.size2() == outputSize());
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		
		RealMatrix outputDelta = coefficients;
		RealMatrix hiddenDelta;
		computeDelta(state,outputDelta,hiddenDelta);
		computeParameterDerivative(patterns,outputDelta,hiddenDelta,state,gradient);
	}
	
	void weightedInputDerivative(
		BatchInputType const& patterns, RealMatrix const& coefficients, State const& state, BatchInputType& inputDerivative
	)const{
		SIZE_CHECK(coefficients.size2() == outputSize());
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		
		RealMatrix outputDelta = coefficients;
		RealMatrix hiddenDelta;
		computeDelta(state,outputDelta,hiddenDelta,inputDerivative);
	}
	
	virtual void weightedDerivatives(
		BatchInputType const & patterns,
		BatchOutputType const & coefficients,
		State const& state,
		RealVector& parameterDerivative,
		BatchInputType& inputDerivative
	)const{
		SIZE_CHECK(coefficients.size2() == outputSize());
		SIZE_CHECK(coefficients.size1() == patterns.size1());

		RealMatrix outputDelta = coefficients;
		RealMatrix hiddenDelta;
		computeDelta(state,outputDelta,hiddenDelta,inputDerivative);
		computeParameterDerivative(patterns,outputDelta,hiddenDelta,state,parameterDerivative);
	}
	
	void setStructure(
		std::size_t in,std::size_t hidden
	){
		m_weightMatrix.resize(hidden,in);
		m_hiddenBias.resize(hidden);
		m_outputBias.resize(in);
	}
	
	//! From ISerializable, reads a model from an archive
	void read( InArchive & archive ){
		archive>>m_weightMatrix;
		archive>>m_hiddenBias;
		archive>>m_outputBias;
	}

	//! From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const{
		archive<<m_weightMatrix;
		archive<<m_hiddenBias;
		archive<<m_outputBias;
	}


private:
	
	void computeDelta(
		State const& state, RealMatrix& outputDelta, RealMatrix& hiddenDelta
	)const{
		InternalState const& s = state.toState<InternalState>();

		noalias(outputDelta) *= m_outputNeuron.derivative(s.outputResponses);
		hiddenDelta.resize(outputDelta.size1(),numberOfHiddenNeurons());
		axpy_prod(outputDelta,decoderMatrix(),hiddenDelta,true);
		noalias(hiddenDelta) *= m_hiddenNeuron.derivative(s.hiddenResponses);
	}
	
	void computeDelta(
		State const& state, RealMatrix& outputDelta, RealMatrix& hiddenDelta, RealMatrix& inputDelta
	)const{
		computeDelta(state,outputDelta,hiddenDelta);
		inputDelta.resize(outputDelta.size1(),inputSize());
		axpy_prod(hiddenDelta,encoderMatrix(),inputDelta,true);
	}
	
	void computeParameterDerivative(
		RealMatrix const& patterns, RealMatrix const& outputDelta, RealMatrix const& hiddenDelta,
		State const& state, RealVector& gradient
	)const{
		InternalState const& s = state.toState<InternalState>();
		std::size_t hiddenParams = inputSize()*numberOfHiddenNeurons();
		std::size_t numHidden = numberOfHiddenNeurons();
		gradient.resize(numberOfParameters());
		gradient.clear();
		axpy_prod(
			trans(s.hiddenResponses),
			outputDelta,
			to_matrix(subrange(gradient,0,hiddenParams),numHidden,inputSize()),false
		);
		axpy_prod(
			trans(hiddenDelta),
			patterns,
			to_matrix(subrange(gradient,0,hiddenParams),numHidden,inputSize()),false
		);
		
		std::size_t hiddenBiasPos = hiddenParams;
		std::size_t outputBiasPos = hiddenBiasPos+numHidden;
		subrange(gradient,hiddenBiasPos,outputBiasPos) = sum_rows(hiddenDelta);
		subrange(gradient,outputBiasPos,outputBiasPos+inputSize()) = sum_rows(outputDelta);
	}

	//! weight matrix between input and hidden layer. the transpose of this is used to connect hidden->output.
	RealMatrix m_weightMatrix;
	//! bias weights of the hidden neurons
	RealVector m_hiddenBias;
	//! bias weights of the visible neurons
	RealVector m_outputBias;

	//!Type of hidden neuron. See Models/Neurons.h for a few choices
	HiddenNeuron m_hiddenNeuron;
	//! Type of output neuron. See Models/Neurons.h for a few choices
	OutputNeuron m_outputNeuron;
};


}
#endif
