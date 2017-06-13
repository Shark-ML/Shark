/*!
 * \brief   Implements the autoencoder with tied weights
 * 
 * \author      O. Krause
 * \date        2010-2017
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
#ifndef SHARK_MODELS_TIEDAUTOENCODER_H
#define SHARK_MODELS_TIEDAUTOENCODER_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/NeuronLayers.h>

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
		typename HiddenNeuron::State hiddenState;
		typename OutputNeuron::State outputState;
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
		return to_vector(m_weightMatrix) | m_hiddenBias | m_outputBias;
	}
	//! uses the values inside the parametervector to set the used values inside the weight matrix
	void setParameterVector(RealVector const& newParameters){
		SIZE_CHECK(newParameters.size() == numberOfParameters());
		std::size_t endWeights = m_weightMatrix.size1() * m_weightMatrix.size2();
		std::size_t endBias = endWeights + m_hiddenBias.size();
		noalias(to_vector(m_weightMatrix)) = subrange(newParameters,0,endWeights);
		noalias(m_hiddenBias) = subrange(newParameters,endWeights,endBias);
		noalias(m_outputBias) = subrange(newParameters,endBias,endBias+m_outputBias.size());
	}
	
	/// \brief Returns the activation function of the hidden units.
	HiddenNeuron const& hiddenActivationFunction()const{
		return m_hiddenNeurons;
	}
	/// \brief Returns the activation function of the output units.
	OutputNeuron const& outputActivationFunction()const{
		return m_outputNeurons;
	}
	
	/// \brief Returns the activation function of the hidden units.
	HiddenNeuron& hiddenActivationFunction(){
		return m_hiddenNeurons;
	}
	/// \brief Returns the activation function of the output units.
	OutputNeuron& outputActivationFunction(){
		return m_outputNeurons;
	}

	//! \brief Returns the output of all neurons after the last call of eval
	//!
	//!     \param  state last result of eval
	//!     \return Output value of the neurons.
	RealMatrix const& hiddenResponses(State const& state)const{
		InternalState const& s = state.toState<InternalState>();
		return s.hiddenResponses;
	}
	
	//! \brief Returns the stored state of the hidden neurons  after the last call of eval
	//!
	//! This method is needed to compute derivatives of the neurons
	//! \param  state last result of eval
	//! \return Output value of the neurons.
	typename HiddenNeuron::State const& hiddenState(State const& state)const{
		InternalState const& s = state.toState<InternalState>();
		return s.hiddenState;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}
	
	Data<RealVector> encode(Data<RealVector> const& patterns)const{
		SHARK_RUNTIME_CHECK(dataDimension(patterns) == inputSize(), "data has not the right input dimensionality");
		int batches = (int) patterns.numberOfBatches();
		Data<RealVector> result(batches);
		SHARK_PARALLEL_FOR(int i = 0; i < batches; ++i){
			auto& inputs = patterns.batch(i);
			auto& outputs = result.batch(i);
			outputs.resize(inputs.size1(),numberOfHiddenNeurons());
			noalias(outputs) = patterns % trans(encoderMatrix()) + repeat(hiddenBias(),inputs.size1());
			m_hiddenNeurons.evalInPlace(outputs);
		}
		return result;
	}
	
	Data<RealVector> decode(Data<RealVector> const& patterns)const{
		SHARK_RUNTIME_CHECK(dataDimension(patterns) == numberOfHiddenNeurons(), "data has not the right input dimensionality");
		int batches = (int) patterns.numberOfBatches();
		Data<RealVector> result(batches);
		SHARK_PARALLEL_FOR(int i = 0; i < batches; ++i){
			auto& inputs = patterns.batch(i);
			auto& outputs = result.batch(i);
			outputs.resize(inputs.size1(),outputSize());
			noalias(outputs) = patterns % trans(decoderMatrix()) + repeat(outputBias(),inputs.size1());
			m_outputNeurons.evalInPlace(outputs);
		}
		return result;
	}
	
	template<class Label>
	LabeledData<RealVector,Label> encode(
		LabeledData<RealVector,Label> const& data
	)const{
		return LabeledData<RealVector,Label>(encode(data.inputs()),data.labels());
	}
	
	template<class Label>
	LabeledData<RealVector,Label> decode(
		LabeledData<RealVector,Label> const& data
	)const{
		return LabeledData<RealVector,Label>(decode(data.inputs()),data.labels());
	}
	
	void eval(RealMatrix const& patterns,RealMatrix& outputs, State& state)const{
		SIZE_CHECK(patterns.size2() == inputSize());
		InternalState& s = state.toState<InternalState>();
			
		std::size_t numPatterns = patterns.size1();
		s.hiddenResponses.resize(numPatterns,numberOfHiddenNeurons());
		noalias(s.hiddenResponses) = patterns % trans(encoderMatrix()) + repeat(hiddenBias(),numPatterns);
		m_hiddenNeurons.evalInPlace(s.hiddenResponses, s.hiddenState);
		
		outputs.resize(numPatterns,outputSize());
		noalias(outputs) = s.hiddenResponses % trans(decoderMatrix()) + repeat(outputBias(),numPatterns);
		m_outputNeurons.evalInPlace(outputs,s.outputState);
	}
	using AbstractModel<RealVector,RealVector>::eval;

	void weightedParameterDerivative(
		BatchInputType const& patterns, RealMatrix const& outputs,  
		RealMatrix const& coefficients, State const& state, RealVector& gradient
	)const{
		SIZE_CHECK(coefficients.size2() == outputSize());
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		
		RealMatrix outputDelta;
		RealMatrix hiddenDelta;
		computeDelta(state, outputs, coefficients, outputDelta,hiddenDelta);
		computeParameterDerivative(patterns,outputDelta,hiddenDelta,state,gradient);
	}
	
	void weightedInputDerivative(
		BatchInputType const& patterns, RealMatrix const& outputs,  
		RealMatrix const& coefficients, State const& state, BatchInputType& inputDerivative
	)const{
		SIZE_CHECK(coefficients.size2() == outputSize());
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		
		RealMatrix outputDelta;
		RealMatrix hiddenDelta;
		computeDelta(state, outputs, coefficients, outputDelta,hiddenDelta,inputDerivative);
	}
	
	virtual void weightedDerivatives(
		BatchInputType const & patterns,
		RealMatrix const& outputs, 
		BatchOutputType const & coefficients,
		State const& state,
		RealVector& parameterDerivative,
		BatchInputType& inputDerivative
	)const{
		SIZE_CHECK(coefficients.size2() == outputSize());
		SIZE_CHECK(coefficients.size1() == patterns.size1());

		RealMatrix outputDelta;
		RealMatrix hiddenDelta;
		computeDelta(state, outputs, coefficients, outputDelta,hiddenDelta,inputDerivative);
		computeParameterDerivative(patterns, outputDelta,hiddenDelta,state,parameterDerivative);
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
		State const& state, RealMatrix const& outputs, RealMatrix const& coefficients, RealMatrix& outputDelta, RealMatrix& hiddenDelta
	)const{
		InternalState const& s = state.toState<InternalState>();
		outputDelta.resize(coefficients.size1(), coefficients.size2());
		noalias(outputDelta) = coefficients;
		m_outputNeurons.multiplyDerivative(outputs,outputDelta,s.outputState);
		
		hiddenDelta.resize(outputDelta.size1(),numberOfHiddenNeurons());
		noalias(hiddenDelta) = outputDelta % decoderMatrix();
		m_hiddenNeurons.multiplyDerivative(s.hiddenResponses,hiddenDelta,s.hiddenState);
		
	}
	
	void computeDelta(
		State const& state, RealMatrix const& outputs, RealMatrix const& coefficients, RealMatrix& outputDelta, RealMatrix& hiddenDelta, RealMatrix& inputDelta
	)const{
		computeDelta(state, outputs, coefficients, outputDelta,hiddenDelta);
		inputDelta.resize(outputDelta.size1(),inputSize());
		noalias(inputDelta) = hiddenDelta % encoderMatrix();
	}
	
	void computeParameterDerivative(
		RealMatrix const& patterns, RealMatrix const& outputDelta, RealMatrix const& hiddenDelta,
		State const& state, RealVector& gradient
	)const{
		InternalState const& s = state.toState<InternalState>();
		std::size_t hiddenParams = inputSize() * numberOfHiddenNeurons();
		std::size_t numHidden = numberOfHiddenNeurons();
		gradient.resize(numberOfParameters());
		auto  gradEncoder  = to_matrix(subrange(gradient,0,hiddenParams),numHidden,inputSize());
		noalias(gradEncoder) = trans(s.hiddenResponses) % outputDelta;
		noalias(gradEncoder) += trans(hiddenDelta) % patterns;
		
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
	HiddenNeuron m_hiddenNeurons;
	//! Type of output neuron. See Models/Neurons.h for a few choices
	OutputNeuron m_outputNeurons;
};


}
#endif
