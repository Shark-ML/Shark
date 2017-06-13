/*!
 * \brief   Implements the autoencoder
 * 
 * \author      O. Krause
 * \date        2010-2014
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
#ifndef SHARK_MODELS_AUTOENCODER_H
#define SHARK_MODELS_AUTOENCODER_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/ConcatenatedModel.h>
#include <shark/Models/LinearModel.h>
#include <shark/Models/NeuronLayers.h>

namespace shark{

/// \brief implements the autoencoder
///
/// The formula is
///  \f[ f(x) = \sigma_2(W^T\sigma_1(Wx+b_1)+b_2)\f]
/// Where \f$ W, W_2, b_1 \f$ and \f$b_2 \f$ are the weights and
///  \f$\sigma_1\f$ and \f$ \sigma_2\f$ are the activation functions for hidden and output units.
///
/// see TiedAutoencoder for the tied weights version where \f$ W_2=W_1^T \f$.
template<class HiddenNeuron,class OutputNeuron>
class Autoencoder :public AbstractModel<RealVector,RealVector>
{
public:
	Autoencoder()
	: m_fullNetwork(m_encoder >> m_decoder ){
		m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
		m_features|=HAS_FIRST_INPUT_DERIVATIVE;
	}

	//! \brief From INameable: return the class name.
	std::string name() const{ 
		return "Autoencoder";
	}

	//! \brief Number of input neurons.
	std::size_t inputSize() const{
		return m_encoder.inputSize();
	}
	//! \brief Number of output neurons.
	std::size_t outputSize() const{
		return m_decoder.outputSize();
	}
	
	//! \brief Total number of hidden neurons.
	std::size_t numberOfHiddenNeurons()const{
		return m_encoder.outputSize();
	}
	
	/// \brief Returns the hidden bias weight vector.
	RealVector const& hiddenBias()const{
		return m_encoder.offset();
	}
	
	/// \brief Returns the hidden bias weight vector.
	RealVector& hiddenBias(){
		return m_encoder.offset();
	}
	
	/// \brief Returns the output bias weight vector.
	RealVector const& outputBias()const{
		return m_decoder.offset();
	}
	/// \brief Returns the output bias weight vector.
	RealVector& outputBias(){
		return m_decoder.offset();
	}
	
	/// \brief Weight matrix for the direction input->hidden.
	RealMatrix const& encoderMatrix()const{
		return m_encoder.matrix();
	}
	/// \brief Weight matrix for the direction input->hidden.
	RealMatrix& encoderMatrix(){
		return m_encoder.matrix();
	}
	/// \brief Weight matrix for the direction hidden->output
	///
	RealMatrix const& decoderMatrix()const{
		return m_decoder.matrix();
	}
	/// \brief Weight matrix for the direction hidden->output
	RealMatrix& decoderMatrix(){
		return m_decoder.matrix();
	}
	
	//! \brief Returns the total number of parameters of the network. 
	std::size_t numberOfParameters()const{
		return m_encoder.numberOfParameters() + m_decoder.numberOfParameters();
	}

	//! returns the vector of used parameters inside the weight matrix
	RealVector parameterVector() const{
		return m_encoder.parameterVector() | m_decoder.parameterVector();
	}
	//! uses the values inside the parametervector to set the used values inside the weight matrix
	void setParameterVector(RealVector const& newParameters){
		SIZE_CHECK(newParameters.size() == numberOfParameters());
		
		std::size_t endWeights1 = m_encoder.numberOfParameters();
		m_encoder.setParameterVector(subrange(newParameters,0,endWeights1));
		m_decoder.setParameterVector(subrange(newParameters,endWeights1,newParameters.size()));
	}

	//! \brief Returns the output of all neurons after the last call of eval
	//!
	//!     \param  state last result of eval
	//!     \return Output value of the neurons.
	RealMatrix const& hiddenResponses(State const& state)const{
		return m_fullNetwork.hiddenResponses(state,0);
	}
	
	//! \brief Returns the stored state of the hidden neurons  after the last call of eval
	//!
	//! This method is needed to compute derivatives of the neurons
	//! \param  state last result of eval
	//! \return Output value of the neurons.
	typename HiddenNeuron::State const& hiddenState(State const& state)const{
		//this hack uses knowledge of that the state of the linear model is the same as the state of the neuron
		//it stores
		return m_fullNetwork.hiddenState(state,0).toState<typename HiddenNeuron::State>();
	}
	
	
	/// \brief Returns the activation function of the hidden units.
	HiddenNeuron const& hiddenActivationFunction()const{
		return m_encoder.activationFunction();
	}
	/// \brief Returns the activation function of the output units.
	OutputNeuron const& outputActivationFunction()const{
		return m_decoder.activationFunction();
	}
	
	/// \brief Returns the activation function of the hidden units.
	HiddenNeuron& hiddenActivationFunction(){
		return m_encoder.activationFunction();
	}
	/// \brief Returns the activation function of the output units.
	OutputNeuron& outputActivationFunction(){
		return m_decoder.activationFunction();
	}
	
	boost::shared_ptr<State> createState()const{
		return m_fullNetwork.createState();
	}
	
	Data<RealVector> encode(Data<RealVector> const& patterns)const{
		return m_encoder(patterns);
	}
	
	Data<RealVector> decode(Data<RealVector> const& patterns)const{
		return m_decoder(patterns);
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
	
	
	void eval(RealMatrix const& patterns,RealMatrix& output, State& state)const{
		m_fullNetwork.eval(patterns,output,state);
	}
	using AbstractModel<RealVector,RealVector>::eval;

	void weightedParameterDerivative(
		BatchInputType const& patterns, BatchOutputType const& outputs,
		RealMatrix const& coefficients, State const& state, RealVector& gradient
	)const{
		SIZE_CHECK(coefficients.size2() == outputSize());
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		m_fullNetwork.weightedParameterDerivative(patterns,outputs, coefficients, state, gradient);
	}
	
	void weightedInputDerivative(
		BatchInputType const& patterns, BatchOutputType const& outputs, 
		RealMatrix const& coefficients, State const& state, BatchInputType& inputDerivative
	)const{
		SIZE_CHECK(coefficients.size2() == outputSize());
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		
		m_fullNetwork.weightedInputDerivative(patterns,outputs, coefficients, state, inputDerivative);
	}
	
	virtual void weightedDerivatives(
		BatchInputType const & patterns,
		BatchOutputType const & outputs,
		BatchOutputType const & coefficients,
		State const& state,
		RealVector& parameterDerivative,
		BatchInputType& inputDerivative
	)const{
		SIZE_CHECK(coefficients.size2() == outputSize());
		SIZE_CHECK(coefficients.size1() == patterns.size1());

		m_fullNetwork.weightedDerivatives(patterns,outputs, coefficients, state, parameterDerivative, inputDerivative);
	}
	
	void setStructure(
		std::size_t in,std::size_t hidden
	){
		m_encoder.setStructure(in,hidden, true);
		m_decoder.setStructure(hidden,in, true);
	}
	
	//! From ISerializable, reads a model from an archive
	void read( InArchive & archive ){
		archive >> m_encoder;
		archive >> m_decoder;
	}

	//! From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const{
		archive << m_encoder;
		archive << m_decoder;
	}


private:
	LinearModel<RealVector, HiddenNeuron> m_encoder;
	LinearModel<RealVector, OutputNeuron> m_decoder;

	ConcatenatedModel<RealVector> m_fullNetwork;
};


}
#endif
