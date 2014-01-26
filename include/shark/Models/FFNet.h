/*!
 * 
 *
 * \brief       Implements a Feef-Forward multilayer perceptron
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010-2011
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
#ifndef SHARK_MODELS_FFNET_H
#define SHARK_MODELS_FFNET_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/Neurons.h>
#include <boost/serialization/vector.hpp>

namespace shark{

//! \brief Offers the functions to create and to work with a feed-forward network.
template<class HiddenNeuron,class OutputNeuron>
class FFNet :public AbstractModel<RealVector,RealVector>
{
	
	struct InternalState: public State{
		/*!
		 *  \brief Used to store the current results of the activation
		 *         function for all neurons for the last batch of patterns \f$x\f$.
		 *
		 *  There is one value for input+hidden+output units for every element of the batch.
		 *  For every value, the following holds:
		 *  Given a network with \f$M\f$ neurons, including
		 * \f$c\f$ input and \f$n\f$ output neurons the single
		 * values for \f$z\f$ are given as:
		 * <ul>
		 *     <li>\f$z_i = x_i,\ \mbox{for\ } 0 \leq i < c\f$</li>
		 *     <li>\f$z_i = g_{hidden}(x),\ \mbox{for\ } c \leq i < M - n\f$</li>
		 *     <li>\f$z_i = y_{i-M+n} = g_{output}(x),\ \mbox{for\ } M - n \leq
		 *                  i < M\f$</li>
		 * </ul>
		 *
		 */
		RealMatrix responses;
		
		void resize(std::size_t neurons, std::size_t patterns){
			responses.resize(neurons,patterns);
		}
	};
	

public:
	//! Creates an empty feed-forward network. After the constructor is called,
	//! one version of the #setStructure methods or configure needs to be called
	//! to define the network topology.
	FFNet()
	:m_numberOfNeurons(0),m_numberOfParameters(0),m_inputNeurons(0),m_outputNeurons(0),m_biasNeuron(0),m_firstOutputNeuron(0){
		m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "FFNet"; }

	std::size_t inputSize()const{
		return m_inputNeurons;
	}
	std::size_t outputSize()const{
		return m_outputNeurons;
	}
	std::size_t numberOfNeurons()const{
		return m_numberOfNeurons;
	}
	std::size_t numberOfParameters()const{
		return m_numberOfParameters;
	}
	//! Returns the current connectionMatrix.
	IntMatrix const& connections()const{
		return m_connectionMatrix;
	}
	
	//! Returns whether a connection between from neuron j to neuron i exists.
	bool connection(std::size_t i, std::size_t j)const{
		return m_connectionMatrix(i,j);
	}

	///returns the matrices for every layer used by eval
	const std::vector<RealMatrix>& layerMatrices()const{
		return m_layerMatrix;
	}

	std::vector<RealMatrix>& layerMatrices(){
		return m_layerMatrix;
	}

	///returns the matrices for every layer used by backpropagation
	const std::vector<RealMatrix>& backpropMatrices()const{
		return m_backpropMatrix;
	}

	const RealVector& bias()const{
		return m_bias;
	}
	RealVector& bias(){
		return m_bias;
	}

	//! returns the vector of used parameters inside the weight matrix
	RealVector parameterVector() const{
		RealVector parameters(numberOfParameters());
		std::size_t layer = 0;
		std::size_t startNeuron = m_inputNeurons;
		for (size_t i = m_inputNeurons; i < m_connectionMatrix.size1(); i++){
			//check whether we have to change layers
			if(i >= startNeuron + m_layerMatrix[layer].size1()){
				startNeuron+= m_layerMatrix[layer].size1();
				++layer;
			}
			//only check weights which are in the layer
			for (size_t j = startNeuron - m_layerMatrix[layer].size2(); j < i; j++){
				if (m_connectionMatrix(i, j) ){
					//find the weight of the forward propagation matrix
					std::size_t indexI = i - startNeuron;
					std::size_t indexJ = j + m_layerMatrix[layer].size2()-startNeuron;

					//use the stored parameter number in the fields of connection Matrix
					//to get the position in the parameter vector
					parameters(m_connectionMatrix(i, j)-1) = m_layerMatrix[layer](indexI, indexJ);
				}
			}
		}
		//write bias neurons if necessary
		for (size_t i = m_inputNeurons; i < m_connectionMatrix.size1() && m_biasNeuron; i++){
			if (m_connectionMatrix(i, m_biasNeuron)){
				parameters(m_connectionMatrix(i, m_biasNeuron)-1) = m_bias(i);
			}
		}
		return parameters;
	}
	//! uses the values inside the parametervector to set the used values inside the weight matrix
	void setParameterVector(RealVector const& newParameters){
		std::size_t layer = 0;
		std::size_t startNeuron = m_inputNeurons;
		for (size_t i = m_inputNeurons; i < m_connectionMatrix.size1(); i++){
			//check whether we have to change layers
			if(i >= startNeuron + m_layerMatrix[layer].size1()){
				startNeuron+= m_layerMatrix[layer].size1();
				++layer;
			}
			//only check weights which are in the layer
			for (size_t j = startNeuron - m_layerMatrix[layer].size2(); j < i; j++){
				if (m_connectionMatrix(i, j) ){
					//find the weight of the forward propagation matrix
					std::size_t indexI = i - startNeuron;
					std::size_t indexJ = j + m_layerMatrix[layer].size2()-startNeuron;

					//use the stored parameter number in the fields of connection Matrix
					//to get the position in the parameter vector
					m_layerMatrix[layer](indexI, indexJ) = newParameters(m_connectionMatrix(i, j)-1) ;
				}
			}
		}
		//write bias neurons if necessary
		for (size_t i = m_inputNeurons; i < m_connectionMatrix.size1() && m_biasNeuron; i++){
			if (m_connectionMatrix(i, m_biasNeuron)){
				m_bias(i) = newParameters(m_connectionMatrix(i, m_biasNeuron)-1);
			}
		}
		//we also have to update the backpropagation weights
		layer = 0;
		std::size_t endNeuron = m_inputNeurons;
		startNeuron = 0;
		for (size_t j = 0; j < m_firstOutputNeuron; j++){
			if( j >= endNeuron ){
				++layer;
				startNeuron = endNeuron;
				endNeuron+= m_backpropMatrix[layer].size1();
			}
			for (size_t i = endNeuron; i < endNeuron+m_backpropMatrix[layer].size2(); i++){
				if (m_connectionMatrix(i, j)){
					//find the weight of the backpropagation matrix
					std::size_t indexI = i - endNeuron;
					std::size_t indexJ = j - startNeuron;

					//use the stored parameter number in the fields of connection Matrix
					//to get the position in the parameter vector
					m_backpropMatrix[layer](indexJ,indexI) = newParameters(m_connectionMatrix(i, j)-1);
				}
			}
		}
	}

	//! \brief Returns the output of all neurons after the last call of eval
	//!
	//!     \param  state last result of eval
	//!     \return Output value of the neurons.
	RealMatrix const& neuronResponses(State const& state)const{
		InternalState const& s = state.toState<InternalState>();
		return s.responses;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}

	void eval(RealMatrix const& patterns,RealMatrix& output, State& state)const{
		InternalState& s = state.toState<InternalState>();
		std::size_t numPatterns=patterns.size1();
		//initialize the input layer using the patterns.
		s.resize(m_numberOfNeurons,numPatterns);
		s.responses.clear();
		noalias(subrange(s.responses,0,m_inputNeurons,0,numPatterns))=trans(patterns);
		std::size_t beginNeuron = m_inputNeurons;
		
		for(std::size_t layer = 0; layer != m_layerMatrix.size();++layer){
			const RealMatrix& weights = m_layerMatrix[layer];
			//number of rows of the layer is also the number of neurons
			std::size_t endNeuron = beginNeuron + weights.size1();
			//some subranges of vectors
			//inputs are the last n neurons, where n is the number of columns of the matrix
			const RealSubMatrix input = subrange(s.responses,beginNeuron - weights.size2(),beginNeuron,0,numPatterns);
			//the bias of the layer
			ConstRealVectorRange bias = subrange(m_bias,beginNeuron,endNeuron);
			//the neurons responses
			RealSubMatrix responses = subrange(s.responses,beginNeuron,endNeuron,0,numPatterns);

			//calculate activation. if this is the last layer, use output neuron response instead
			axpy_prod(weights,input,responses);
			if(m_biasNeuron){
				noalias(responses) += trans(repeat(bias,numPatterns));
			}
 			if(layer < m_layerMatrix.size()-1) {
				noalias(responses) = m_hiddenNeuron(responses);
 			}
 			else {
				noalias(responses) = m_outputNeuron(responses);
 			}
			//go to the next layer
			beginNeuron = endNeuron;
		}
		//Sanity check
		SIZE_CHECK(beginNeuron == m_numberOfNeurons);

		//copy output layer into output
		output.resize(numPatterns,m_outputNeurons);
		noalias(output) = subrange(trans(s.responses),0,numPatterns,m_firstOutputNeuron,m_firstOutputNeuron+m_outputNeurons);
	}
	using AbstractModel<RealVector,RealVector>::eval;

	void weightedParameterDerivative(
		RealMatrix const& patterns, RealMatrix const& coefficients, State const& state, RealVector& gradient
	)const{
		SIZE_CHECK(coefficients.size2() == m_outputNeurons);
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		std::size_t numPatterns=patterns.size1();
		
		//initialize delta using coefficients and clear the rest
		RealMatrix delta(m_numberOfNeurons,numPatterns);
		delta.clear();
		RealSubMatrix outputDelta = rows(delta,m_firstOutputNeuron,m_numberOfNeurons);
		noalias(outputDelta) = trans(coefficients);

		//reduce to general case.
		weightedParameterDerivativeFullDelta(patterns,delta,state,gradient);
	}
	
	///\brief Calculates the derivtive for the special case, when error terms for all neurons of the network exist.
	///
	///This is usefull when the hidden neurons need to meet additional requirements.
	///The value of delta is changed during computation and holds the results of the backpropagation steps.
	///The format is such that the rows of delta are the neurons and the columns the patterns.
	void weightedParameterDerivativeFullDelta(
		RealMatrix const& patterns, RealMatrix& delta, State const& state, RealVector& gradient
	)const{
		InternalState const& s = state.toState<InternalState>();
		SIZE_CHECK(delta.size1() >= m_numberOfNeurons-m_inputNeurons);
		SIZE_CHECK(delta.size2() == patterns.size1());
		SIZE_CHECK(s.responses.size2() == patterns.size1());
		std::size_t numPatterns=patterns.size1();

		//initialize output neurons using coefficients
		RealSubMatrix outputDelta = subrange(delta,m_firstOutputNeuron,m_numberOfNeurons,0,numPatterns);
		ConstRealSubMatrix outputResponse = subrange(s.responses,m_firstOutputNeuron,m_numberOfNeurons,0,numPatterns);
		noalias(outputDelta) *= m_outputNeuron.derivative(outputResponse);

		//iterate backwards using the backprop matrix and propagate the errors to get the needed delta values
		std::size_t endNeuron = m_firstOutputNeuron;
		//for the parameter derivative, we don't need the deltas of the input neurons
		for(std::size_t layer = m_backpropMatrix.size()-1; layer > 0; --layer){
			RealMatrix const& weights = m_backpropMatrix[layer];
			std::size_t beginNeuron = endNeuron - weights.size1();

			//get the delta and response values of this layer
			RealSubMatrix layerDelta = subrange(delta,beginNeuron,endNeuron,0,numPatterns);
			RealSubMatrix layerDeltaInput = subrange(delta,endNeuron,endNeuron+weights.size2(),0,numPatterns);
			ConstRealSubMatrix layerResponse = subrange(s.responses,beginNeuron,endNeuron,0,numPatterns);
			RealMatrix propagate(weights.size1(),numPatterns);
			axpy_prod(weights,layerDeltaInput,layerDelta,false);//add the values to the maybe non-empty delta part
			noalias(layerDelta) *= m_hiddenNeuron.derivative(layerResponse);
			//go a layer backwards
			endNeuron=beginNeuron;
		}
		//Sanity check
		SIZE_CHECK(endNeuron == m_inputNeurons);

		// calculate error gradient
		//todo: take network structure into account to prevent checking all possible weights...
		gradient.resize(numberOfParameters());
		size_t pos = 0;
		for (size_t neuron = m_inputNeurons; neuron < m_numberOfNeurons; neuron++){
			for (size_t j = 0; j < neuron; j++){
				if (m_connectionMatrix(neuron, j)){
					gradient(pos) = inner_prod(row(delta,neuron),row(s.responses,j));
					pos++;
				}
			}
		}
		//check whether we need the bias derivative
		if(!m_biasNeuron)
			return;
		//calculate bias derivative
		for (size_t neuron = m_inputNeurons; neuron < m_numberOfNeurons; neuron++){
			if (m_connectionMatrix(neuron, m_biasNeuron)){
				gradient(pos) = sum(row(delta,neuron));
				pos++;
			}
		}
		//Sanity check
		SIZE_CHECK(pos == gradient.size());
	}
	
	//! Based on a given #m_connectionMatrix the structure of the
	//! current network is modified.
	void setStructure(std::size_t in, std::size_t out,IntMatrix const& cmat){
		SIZE_CHECK(cmat.size1()+1 == cmat.size2());
		SIZE_CHECK(cmat.size1() >= in+out);
		m_inputNeurons = in;
		m_outputNeurons = out;
		m_numberOfNeurons = cmat.size1();
		m_firstOutputNeuron=m_numberOfNeurons-m_outputNeurons;
		m_biasNeuron=m_numberOfNeurons;

		//check for bias connections
		m_biasNeuron = 0;
		for(std::size_t i = 0; i!= m_numberOfNeurons;++i){
			if(cmat(i,m_numberOfNeurons)){
				m_biasNeuron = m_numberOfNeurons;
				m_bias.resize(m_numberOfNeurons);
				break;
			}
		}

		//we are allowed to change the values of cmat and we will use that to our advantage. we will store the position
		//of the weight in the parameter vector +1 in there
		std::size_t parameterNum = 1;
		m_connectionMatrix.resize(cmat.size1(),cmat.size2());
		m_connectionMatrix.clear();
		for(std::size_t i = 0; i != cmat.size1(); ++i){
			for(std::size_t j = 0; j != i; ++j){
				if(cmat(i,j)){
					m_connectionMatrix(i,j)=parameterNum;
					++parameterNum;
				}
			}
		}
		//set parameters of bias if necessary
		if(m_biasNeuron){
			for(std::size_t i = 0; i != cmat.size1(); ++i){
				if(cmat(i,m_biasNeuron)){
					m_connectionMatrix(i,m_biasNeuron)=parameterNum;
					++parameterNum;
				}
			}
		}

		//given the connection matrix, we have to find out, how the layers of the matrix are structured
		//every set of neurons i,i+1,...,j-1,j forms a layer if they don't have connections with each other but connections
		//with at least neuron i-1 and j+1. one can check that iteratively by

		std::vector<std::size_t> layerSizes;
		layerSizes.push_back(in);//we allready know the input layer
		std::size_t currentLayerSize = 0;
		std::size_t layerStart=m_inputNeurons;
		//we also know the output layer, thus we don't iterate over these neurons
		for(std::size_t neuron = m_inputNeurons; neuron != m_firstOutputNeuron; ++neuron){
			for(std::size_t i = neuron; i >= layerStart; --i){
				if(cmat(neuron,i) != 0){
					layerSizes.push_back(currentLayerSize);
					layerStart += currentLayerSize;
					currentLayerSize = 0;
					break;
				}
			}
			++currentLayerSize;
		}
		if(currentLayerSize != 0){
			layerSizes.push_back(currentLayerSize);
		}
		layerSizes.push_back(m_outputNeurons);

		//now that we have the layer sizes, we will also check from how many 
		//neurons they get input. for layered structures this leads to
		//an extreme reduction of computation time later
		std::vector<std::size_t> layerInputSizes(layerSizes.size());
		layerInputSizes[0]=0;
		layerInputSizes[1]=in;//the first hidden layer must take input from  the input neurons...
		layerStart = in+layerSizes[1];
		for(std::size_t layer= 2; layer < layerSizes.size(); ++layer){
			std::size_t inputSize = layerStart;
			std::size_t layerEnd = layerStart + layerSizes[layer];
			//now check for every neuron of the layer, whether it is connected to a previous neuron. once we find the first connection,
			//we can stop and move to the next layer
			for(std::size_t i = 0; i != layerStart;++i){
				for(std::size_t neuron=layerStart; neuron!= layerEnd; ++neuron){
					if(cmat(neuron,i)){
						layerInputSizes[layer]=inputSize;
						goto nextLayerInput; //first goto in my life, but only as labeled break
					}
				}
				inputSize--;
			}
		nextLayerInput:
			layerStart = layerEnd;
		}

		//we will now do the same backward to see, which neuron get's direct backpropagation from another neuron
		std::vector<std::size_t> layerBackpropSizes(layerSizes.size());
		layerStart = 0;
		for(std::size_t layer = 0; layer < layerSizes.size()-1; ++layer){
			std::size_t layerEnd = layerStart + layerSizes[layer];
			std::size_t backpropSize = m_numberOfNeurons - layerEnd;

			//now check backwards for every neuron of the layer, whether it is connected to a following neuron. Once we find the first connection,
			//we can stop and move to the next layer
			for(std::size_t i = m_numberOfNeurons-1; i >= layerEnd;--i){
				for(std::size_t neuron = layerStart; neuron != layerEnd; ++neuron){
					if(cmat(i,neuron)){
						layerBackpropSizes[layer]=backpropSize;
						goto nextLayerBackprop; //second goto in my life, also only as labeled break
					}
				}
				backpropSize--;
			}
		nextLayerBackprop:
			layerStart= layerEnd;
		}

		//now - finally, we can create the matrices
		//forward propagation
		m_layerMatrix.resize(layerSizes.size()-1);
		for(std::size_t layer = 1; layer != layerSizes.size();++layer){
			m_layerMatrix[layer-1] = RealMatrix(layerSizes[layer],layerInputSizes[layer]);
			m_layerMatrix[layer-1].clear();
		}
		//backward propagation
		m_backpropMatrix.resize(layerSizes.size()-1);
		for(std::size_t layer = 0; layer != layerSizes.size()-1;++layer){
			m_backpropMatrix[layer] = RealMatrix(layerSizes[layer],layerBackpropSizes[layer]);
			m_layerMatrix[layer].clear();
		}

		resize();
	}

	/*!
	*  \brief Creates a connection matrix for a network with two
	*         hidden layers.
	*
	*  Automatically creates a connection matrix for a network with
	*  four different layers: An input layer with \em in input neurons,
	*  an output layer with \em out output neurons and two hidden layers
	*  with \em hidden1 and \em hidden2 hidden neurons, respectively.
	*  (Standard) connections can be defined by \em ff_layer,
	*  \em ff_in_out, \em ff_all and \em bias.
	*
	*  \param  in         number of input neurons.
	*  \param  hidden1    number of neurons of the first hidden layer.
	*  \param  hidden2    number of neurons of the second hidden layer.
	*  \param  out        number of output neurons.
	*  \param  ff_layer   if set to \em true, connections from
	*                     each neuron of layer \f$i\f$ to each neuron
	*                     of layer \f$i+1\f$ will be set for all layers.
	*  \param  ff_in_out  if set to \em true, connections from
	*                     all input neurons to all output neurons
	*                     will be set.
	*  \param  ff_all     if set to \em true, connections from all
	*                     neurons of layer \f$i\f$ to all neurons of
	*                     layers \f$j\f$ with \f$j > i\f$ will be set
	*                     for all layers \f$i\f$.
	*  \param  bias       if set to \em true, connections from
	*                     all neurons (except the input neurons)
	*                     to the bias will be set.
	*/
	void setStructure(
		std::size_t in,
		std::size_t hidden1,
		std::size_t hidden2,
		std::size_t out,
		bool ff_layer  = true,
		bool ff_in_out = true,
		bool ff_all    = true,
		bool bias      = true
	){
		std::vector<size_t> layer(4);
		layer[0] = in;
		layer[1] = hidden1;
		layer[2] = hidden2;
		layer[3] = out;
		setStructure(layer, ff_layer, ff_in_out, ff_all, bias);
	}

	/*!
	*  \brief Creates a connection matrix for a network.
	*
	*  Automatically creates a connection matrix with several layers, with
	*  the numbers of neurons for each layer defined by \em layers and
	*  (standard) connections defined by \em ff_layer, \em ff_in_out,
	*  \em ff_all and \em bias.
	*
	*  \param  layers     contains the numbers of neurons for each
	*                     layer of the network.
	*  \param  layer   if set to \em true, connections from
	*                     each neuron of layer \f$i\f$ to each neuron
	*                     of layer \f$i+1\f$ will be set for all layers.
	*  \param  inOut  if set to \em true, connections from
	*                     all input neurons to all output neurons
	*                     will be set.
	*  \param  allShortcuts   if set to \em true, connections from all
	*                     neurons of layer \f$i\f$ to all neurons of
	*                     layers \f$j\f$ with \f$j > i\f$ will be set
	*                     for all layers \f$i\f$.
	*  \param  bias       if set to \em true, connections from
	*                     all neurons (except the input neurons)
	*                     to the bias will be set.
	*/
	void setStructure(
		std::vector<size_t> const&layers,
		bool layer,    // all connections between layers?
		bool inOut,   // shortcuts from in to out?
		bool allShortcuts,   // all shortcuts?
		bool bias //bias neuron?
	){
		// Calculate total number of neurons from the
		// number of neurons per layer:
		m_numberOfNeurons = 0;
		for (size_t k = 0; k < layers.size(); k++)
			m_numberOfNeurons += layers[k];
		m_inputNeurons = layers[0];
		m_outputNeurons = layers.back();
		m_firstOutputNeuron=m_numberOfNeurons-m_outputNeurons;
		m_biasNeuron=m_numberOfNeurons;

		IntMatrix connectionMatrix(m_numberOfNeurons, m_numberOfNeurons+1);
		connectionMatrix.clear();

		// set connections from each neuron of layer i to each
		// neuron of layer i + 1 for all layers:
		if (layer)
		{
			size_t z_pos = layers[0];
			size_t s_pos = 0;
			for (size_t k = 0; k < layers.size() - 1; k++)
			{
				for (size_t row = z_pos; row < z_pos + layers[k + 1]; row++)
					for (size_t column = s_pos; column < s_pos + layers[k]; column++)
						connectionMatrix(row, column) = 1;
				s_pos += layers[k];
				z_pos += layers[k + 1];
			}
		}

		// set connections from all input neurons to all output neurons:
		if (inOut)
		{
			for (size_t row = m_firstOutputNeuron; row < m_numberOfNeurons; row++)
				for (size_t column = 0; column < m_inputNeurons; column++)
					connectionMatrix(row, column) = 1;
		}

		// set connections from all neurons of layer i to
		// all neurons of layers j with j > i for all layers i:
		if (allShortcuts)
		{
			size_t z_pos = layers[0];
			size_t s_pos = 0;
			for (size_t k = 0; k < layers.size() - 1; k++)
			{
				for (size_t row = z_pos; row < z_pos + layers[k + 1]; row++)
					for (size_t column = 0; column < s_pos + layers[k]; column++)
						connectionMatrix(row, column) = 1;
				s_pos += layers[k];
				z_pos += layers[k + 1];
			}
		}

		// set connections from all neurons (except the input neurons)
		// to the bias values:
		if (bias)
		{
			m_biasNeuron=m_numberOfNeurons;
			for (size_t k = m_inputNeurons; k < m_numberOfNeurons; k++)
				connectionMatrix(k, m_biasNeuron) = 1;
		}

		//of course, we allready have layer information in place and thus it is not really efficient to do this.
		//But setStructure won't be called very frequently.
		//Be aware that we actually change the values in the connectionMatrix with this call!
		setStructure(m_inputNeurons,m_outputNeurons,connectionMatrix);
	}
	/*!
	*  \brief Creates a connection matrix for a network with a
	*         single hidden layer
	*
	*  Automatically creates a connection matrix for a network with
	*  three different layers: An input layer with \em in input neurons,
	*  an output layer with \em out output neurons and one hidden layer
	*  with \em hidden neurons, respectively.
	*  (Standard) connections can be defined by \em ff_layer,
	*  \em ff_in_out, \em ff_all and \em bias.
	*
	*  \param  in         number of input neurons.
	*  \param  hidden    number of neurons of the second hidden layer.
	*  \param  out        number of output neurons.
	*  \param  ff_layer   if set to \em true, connections from
	*                     each neuron of layer \f$i\f$ to each neuron
	*                     of layer \f$i+1\f$ will be set for all layers.
	*  \param  ff_in_out  if set to \em true, connections from
	*                     all input neurons to all output neurons
	*                     will be set.
	*  \param  ff_all     if set to \em true, connections from all
	*                     neurons of layer \f$i\f$ to all neurons of
	*                     layers \f$j\f$ with \f$j > i\f$ will be set
	*                     for all layers \f$i\f$.
	*  \param  bias       if set to \em true, connections from
	*                     all neurons (except the input neurons)
	*                     to the bias will be set.
	*/
	void setStructure(
		std::size_t in,
		std::size_t hidden,
		std::size_t out,
		bool ff_layer  = true,
		bool ff_in_out = true,
		bool ff_all    = true,
		bool bias      = true
	){
		std::vector<size_t> layer(3);
		layer[0] = in;
		layer[1] = hidden;
		layer[2] = out;
		setStructure(layer, ff_layer, ff_in_out, ff_all, bias);
	}

	//! \brief Configures the network.
	//!
	//!  The Data format is as follows:
	//!   general properties:
	//!  "inputs" number of input neurons. No default value, must be set!
	//!  "outputs" number of output neurons. No defualt value, must be set!
	//!  "layers" whether connections between adjacent layers should be set. default:true
	//!  "inOutConnections" whether shortcuts between in-andoutput neurons should be set. default:true
	//!  "shortcuts" shortcuts between hidden layers. default:true
	//!  "bias" whether the bias should be acitvated. default:true
	//!
	//! for every hidden Layer a node must be present. The name of the node must be "layer"
	//! and it needs the following properties
	//! "number" the layer number - the first has number 1, the next 2...
	//! "neurons" the number of neurons in the layer.
	//! The count begins at the bottom of the network. The input layer would have number 0.
	//! it is not checked whether the layer numbering makes sense, but it is not allowed that the number
	//! of a layer exceeds the total number of layers.
	void configure(PropertyTree const& node )
	{
		typedef PropertyTree::const_assoc_iterator Iter;
		//general
		size_t inputNeurons = node.get<size_t>("inputs");
		size_t outputNeurons = node.get<size_t>("outputs");
		bool layerConnections = node.get("layers",true);
		bool inOutConnections = node.get("inOutConnections",true);
		bool allConnections = node.get("shortcuts",true);
		bool biasConnections = node.get("bias",true);

		//commented out because of bugs in boost!
		//std::pair<Iter,Iter> range = node.equal_range("layer");
		//instead this piece of code must be used
		std::pair<Iter,Iter> range = std::make_pair(node.ordered_begin(),node.not_found());
		bool first=true;
		bool last=false;
		for(Iter node=range.first;node!=range.second&&!last;++node){
			if(first && node->first=="layer"){
				first = false;
				range.first=node;
			}
			if(!first && node->first!="layer"){
				last = true;
				range.second=node;
			}
		}
		size_t numLayers = std::distance(range.first,range.second);
		if(numLayers){
			std::vector<size_t> layers(numLayers+2);
			layers[0]=inputNeurons;
			layers.back()=outputNeurons;
			for(Iter layer=range.first;layer!=range.second;++layer){
				size_t layerPos = layer->second.get<size_t>("number");
				size_t layerSize = layer->second.get<size_t>("neurons");
				if(layerPos > numLayers)
					SHARKEXCEPTION("[FFNet::configure] layer number too big");
				layers[layerPos+1] = layerSize;
			}
			setStructure(layers,layerConnections,inOutConnections,allConnections,biasConnections);
		}
		else{
			setStructure(inputNeurons,0,outputNeurons,layerConnections,inOutConnections,allConnections,biasConnections);
		}
	}

	/// From ISerializable, reads a model from an archive
	void read( InArchive & archive ){
		archive>>m_inputNeurons;
		archive>>m_outputNeurons;
		archive>>m_connectionMatrix;
		archive>>m_layerMatrix;
		archive>>m_backpropMatrix;
		archive>>m_bias;
		archive>>m_biasNeuron;
		resize();
	}

	/// From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const{
		archive<<m_inputNeurons;
		archive<<m_outputNeurons;
		archive<<m_connectionMatrix;
		archive<<m_layerMatrix;
		archive<<m_backpropMatrix;
		archive<<m_bias;
		archive<<m_biasNeuron;
	}


protected:
	//! Reserves memory for all internal net data structures and updates the internal states.
	void resize(){
		m_numberOfNeurons = m_connectionMatrix.size1();
		m_firstOutputNeuron = m_numberOfNeurons - m_outputNeurons;
		m_numberOfParameters = 0;
		for (size_t i = 0; i < m_numberOfNeurons; i++)
		{
			for (size_t j = 0; j <= i; j++)
				if (m_connectionMatrix(i, j)) m_numberOfParameters++;
			if (m_connectionMatrix(i, m_biasNeuron)) m_numberOfParameters++;
		}
	}

	//!  \brief Number of all network neurons.
	//!
	//!  This is the total number of neurons in the network, i.e.
	//!  input, hidden and output neurons.
	std::size_t m_numberOfNeurons;
	std::size_t m_numberOfParameters;
	std::size_t m_inputNeurons;
	std::size_t m_outputNeurons;



	//! \brief Index of bias weight in the weight matrix.
	//!
	//! The bias term can be considered as an extra input
	//! neuron \f$x_0\f$, whose activation is permanently set to \f$+1\f$.
	//! The index is used for direct access of the weight for the bias
	//! connection in the weight matrix. It is set to zero, if there is no bias
	std::size_t m_biasNeuron;


	//! \brief Index of the first output neuron.
	//!
	//! This value is used for direct access of the output neurons,
	//! especially when working with the list  of activation
	//! results for all neurons.
	std::size_t m_firstOutputNeuron;


	/*! \brief Matrix that identifies the connection status between
	 *         all neurons.
	 *
	 * If the network consists of \f$M\f$ neurons, this is a
	 * \f$M \times M + 1\f$ matrix where a value of "1" at position \f$(i,j)\f$
	 * denotes a connection from neuron \f$j\f$ to
	 * neuron \f$i\f$ with \f$j < i\f$, a value of zero means
	 * no connection. <br>
	 * The last column of the matrix indicates the connection of the
	 * current neuron (given by the row number) to the bias value. <br>
	 * For all input neurons, this value is always zero.
	 * The implementation is allowed to change the values of the matrix. 
	 * It is guaranteed that a value != 0 is a connection between the
	 * neurons. Users should never check for C(i,j)==1
	 *
	 * \f$
	 * C = \left( \begin{array}{cccccc}
	 *         0          & 0          & 0          & \cdots & 0 & c_{0b}\\
	 *         c_{10}     & 0          & 0          & \cdots & 0 & c_{1b}\\
	 *         c_{20}     & c_{21}     & 0          & \cdots & 0 & c_{2b}\\
	 *         \vdots     & \vdots     & \vdots     & \ddots & \vdots & \vdots\\
	 *         c_{(M-1)0} & c_{(M-1)1} & c_{(M-1)2} & \cdots & 0 & c_{(M-1)b}\\
	 * \end{array} \right)\f$
	 *
	 * \sa m_weightMatrix
	 *
	 */
	IntMatrix m_connectionMatrix;

	//!\brief represents the connection matrix using a layered structure for forward propagation
	//!
	//! a layer is made of neurons with consecutive indizes which are not
	//! connected with each other. In other words, if there exists a k i<k<j such
	//! that C(i,k) = 1 or C(k,j) = 1 or C(j,i) = 1 than the neurons i,j are not in the same layer.
	//! This is the forward view, meaning that the layers holds the weights which are used to calculate
	//! the activation of the neurons of the layer.
	std::vector<RealMatrix> m_layerMatrix;
	//!\brief represents the backwards view of the network as layered structure.
	//!
	//! This is the backward view of the Network which is used for the backpropagation step. So every
	//! Matrix contains the weights of the neurons which are activatived by the layer.
	std::vector<RealMatrix> m_backpropMatrix;

	//! bias weights of the neurons
	RealVector m_bias;

	//!Type of hidden neuron. See Models/Neurons.h for a few choice
	HiddenNeuron m_hiddenNeuron;
	//! Type of output neuron. See Models/Neurons.h for a few choice
	OutputNeuron m_outputNeuron;
};

///FFNet with symmetric sigmoids with range [-1,1]
typedef FFNet<TanhNeuron,TanhNeuron> SimpleFFNet;
///FFNet with symmetric sigmoids in the hidden neuron and linear outputs
typedef FFNet<TanhNeuron,LinearNeuron> LinOutFFNet;

}
#endif



