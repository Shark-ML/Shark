/*!
 * 
 *
 * \brief       Implements a Feef-Forward multilayer perceptron
 * 
 * 
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
#ifndef SHARK_MODELS_FFNET_H
#define SHARK_MODELS_FFNET_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/Neurons.h>
#include <boost/serialization/vector.hpp>

namespace shark{
	
struct FFNetStructures{
	enum ConnectionType{
		Normal, //< Layerwise connectivity without shortcuts
		InputOutputShortcut, //< Normal with additional shortcuts from input to output neuron
		Full //< Every layer is fully connected to all neurons in the lower layer 
	};
};

//! \brief Offers the functions to create and to work with a feed-forward network.
//!
//! A feed forward network consists of several layers. every layer consists of a linear 
//! function with optional bias whose response is modified by a (nonlinear) activation function.
//! starting from the input layer, the output of every layer is the input of the next.
//! The two template arguments goveern the activation functions of the network. 
//! The activation functions are typically sigmoidal.
//! All hidden layers share one activation function, while the output layer can be chosen to use
//! a different one, for example to allow the last output to be unbounded, in which case a 
//! linear output function is used.
//! It is not possible to use arbitrary activation functions but Neurons following in the structure
//! in Models/Neurons.h Especially it holds that the derivative of the activation function
//! must have the form f'(x) = g(f(x)).
//!
//! This network class allows for several different topologies of structure. The layerwise structure
//! outlined above is the ddefault one, but the network also allows for shortcuts. most typically 
//! an input-output shotcut is used, that is a shortcut that connects the input neurons directly 
//! with the output using linear weights. But also a fully connected structure is possible, where
//! every layer is fed as input to every successive layer instead of only the next one.
template<class HiddenNeuron,class OutputNeuron>
class FFNet :public AbstractModel<RealVector,RealVector>
{
	struct InternalState: public State{
		//!  \brief Used to store the current results of the activation
		//!         function for all neurons for the last batch of patterns \f$x\f$.
		//!
		//!  There is one value for input+hidden+output units for every element of the batch.
		//!  For every value, the following holds:
		//!  Given a network with \f$M\f$ neurons, including
		//! \f$c\f$ input and \f$n\f$ output neurons the single
		//! values for \f$z\f$ are given as:
		//! <ul>
		//!     <li>\f$z_i = x_i,\ \mbox{for\ } 0 \leq i < c\f$</li>
		//!     <li>\f$z_i = g_{hidden}(x),\ \mbox{for\ } c \leq i < M - n\f$</li>
		//!     <li>\f$z_i = y_{i-M+n} = g_{output}(x),\ \mbox{for\ } M - n \leq
		//!                  i < M\f$</li>
		//! </ul>
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
	:m_numberOfNeurons(0),m_inputNeurons(0),m_outputNeurons(0){
		m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
		m_features|=HAS_FIRST_INPUT_DERIVATIVE;
	}

	//! \brief From INameable: return the class name.
	std::string name() const
	{ return "FFNet"; }

	//! \brief Number of input neurons.
	std::size_t inputSize()const{
		return m_inputNeurons;
	}
	//! \brief Number of output neurons.
	std::size_t outputSize()const{
		return m_outputNeurons;
	}
	//! \brief Total number of neurons, that is inputs+hidden+outputs.
	std::size_t numberOfNeurons()const{
		return m_numberOfNeurons;
	}
	//! \brief Total number of hidden neurons.
	std::size_t numberOfHiddenNeurons()const{
		return numberOfNeurons() - inputSize() -outputSize();
	}

	//! \brief Returns the matrices for every layer used by eval.
	std::vector<RealMatrix> const& layerMatrices()const{
		return m_layerMatrix;
	}

	//! \brief Returns the matrices for every layer used by backpropagation.
	std::vector<RealMatrix> const& backpropMatrices()const{
		return m_backpropMatrix;
	}
	
	//! \brief Returns the direct shortcuts between input and output neurons.
	//!
	//! This does not necessarily exist.
	RealMatrix const& inputOutputShortcut() const{
		return m_inputOutputShortcut;
	}

	//! \brief Returns the bias values for hidden and output units.
	//!
	//! This is either empty or a vector of size numberOfNeurons()-inputSize().
	//! the first entry is the value of the first hidden unit while the last outputSize() units
	//! are the values of the output units.
	const RealVector& bias()const{
		return m_bias;
	}
	
	//! \brief Returns the total number of parameters of the network. 
	std::size_t numberOfParameters()const{
		std::size_t numParams = m_inputOutputShortcut.size1()*m_inputOutputShortcut.size2();
		numParams += bias().size();
		for(std::size_t i = 0; i != layerMatrices().size(); ++i){
			numParams += layerMatrices()[i].size1()*layerMatrices()[i].size2();
		}
		return numParams;
	}

	//! returns the vector of used parameters inside the weight matrix
	RealVector parameterVector() const{
		RealVector parameters(numberOfParameters());
		init(parameters) << matrixSet(m_layerMatrix),m_bias,toVector(m_inputOutputShortcut);
		return parameters;
	}
	//! uses the values inside the parametervector to set the used values inside the weight matrix
	void setParameterVector(RealVector const& newParameters){
		//set the normal forward propagation weights
		init(newParameters) >> matrixSet(m_layerMatrix),m_bias,toVector(m_inputOutputShortcut);
		
		//we also have to update the backpropagation weights
		//this is more or less an inversion. for all connections of a neuron i with a neuron j, i->j
		//the backpropagation matrix has an entry j->i.
		
		// we start with all neurons in layer i, looking at all layers j > i and checking whether 
		// they are connected, in this case we transpose the part of the matrix which is connecting
		// layer j with layer i and copying it into the backprop matrix.
		// we assume here, that either all neurons in layer j are connected to all neurons in layer i
		// or that there are no connections at all beetween the layers.
		std::size_t layeriStart = 0;
		for(std::size_t layeri = 0; layeri != m_layerMatrix.size(); ++layeri){
			std::size_t columni = 0;
			std::size_t neuronsi = inputSize();
			if(layeri > 0)
				neuronsi = m_layerMatrix[layeri-1].size1();
			
			std::size_t layerjStart = layeriStart + neuronsi;
			for(std::size_t layerj = layeri; layerj != m_layerMatrix.size(); ++layerj){
				std::size_t neuronsj = m_layerMatrix[layerj].size1();
				//only process, if layer j has connections with layer i
				if(layerjStart-m_layerMatrix[layerj].size2() <= layeriStart){
					
					//Start of the weight columns to layer i in layer j.
					//parantheses are important to protect against underflow
					std::size_t weightStartj = layeriStart -(layerjStart - m_layerMatrix[layerj].size2());
					noalias(columns(m_backpropMatrix[layeri],columni,columni+neuronsj)) 
					= trans(columns(m_layerMatrix[layerj],weightStartj,weightStartj+neuronsi)); 
				}
				columni += neuronsj;
				layerjStart += neuronsj; 
			}
			layeriStart += neuronsi;
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
		std::size_t numPatterns = patterns.size1();
		//initialize the input layer using the patterns.
		s.resize(numberOfNeurons(),numPatterns);
		s.responses.clear();
		noalias(rows(s.responses,0,m_inputNeurons)) = trans(patterns);
		std::size_t beginNeuron = m_inputNeurons;
		
		for(std::size_t layer = 0; layer != m_layerMatrix.size();++layer){
			const RealMatrix& weights = m_layerMatrix[layer];
			//number of rows of the layer is also the number of neurons
			std::size_t endNeuron = beginNeuron + weights.size1();
			//some subranges of vectors
			//inputs are the last n neurons, where n is the number of columns of the matrix
			RealSubMatrix const input = rows(s.responses,beginNeuron - weights.size2(),beginNeuron);
			//the neurons responses
			RealSubMatrix responses = rows(s.responses,beginNeuron,endNeuron);

			//calculate activation. first compute the linear part and the optional bias and then apply
			// the non-linearity
			axpy_prod(weights,input,responses);
			if(!bias().empty()){
				//the bias of the layer is shifted as input units can not have bias.
				ConstRealVectorRange bias = subrange(m_bias,beginNeuron-inputSize(),endNeuron-inputSize());
				noalias(responses) += trans(repeat(bias,numPatterns));
			}
			// if this is the last layer, use output neuron response instead
 			if(layer < m_layerMatrix.size()-1) {
				noalias(responses) = m_hiddenNeuron(responses);
 			}
 			else {
				//add shortcuts if necessary
				if(m_inputOutputShortcut.size1() != 0){
					axpy_prod(m_inputOutputShortcut,trans(patterns),responses,false);
				}
				noalias(responses) = m_outputNeuron(responses);
 			}
			//go to the next layer
			beginNeuron = endNeuron;
		}
		//Sanity check
		SIZE_CHECK(beginNeuron == m_numberOfNeurons);

		//copy output layer into output
		output.resize(numPatterns,m_outputNeurons);
		noalias(output) = trans(rows(s.responses,m_numberOfNeurons-outputSize(),m_numberOfNeurons));
	}
	using AbstractModel<RealVector,RealVector>::eval;

	void weightedParameterDerivative(
		BatchInputType const& patterns, RealMatrix const& coefficients, State const& state, RealVector& gradient
	)const{
		SIZE_CHECK(coefficients.size2() == m_outputNeurons);
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		std::size_t numPatterns=patterns.size1();
		
		//initialize delta using coefficients and clear the rest. also don't compute the delta for
		// the input nurons as they are not needed.
		RealMatrix delta(numberOfNeurons(),numPatterns,0.0);
		RealSubMatrix outputDelta = rows(delta,delta.size1()-outputSize(),delta.size1());
		noalias(outputDelta) = trans(coefficients);

		computeDelta(delta,state,false);
		computeParameterDerivative(delta,state,gradient);
		
	}
	
	void weightedInputDerivative(
		BatchInputType const& patterns, RealMatrix const& coefficients, State const& state, BatchInputType& inputDerivative
	)const{
		SIZE_CHECK(coefficients.size2() == m_outputNeurons);
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		std::size_t numPatterns=patterns.size1();
		
		//initialize delta using coefficients and clear the rest
		//we compute the full set of delta values here. the delta values of the inputs are the inputDerivative
		RealMatrix delta(numberOfNeurons(),numPatterns,0.0);
		RealSubMatrix outputDelta = rows(delta,delta.size1()-outputSize(),delta.size1());
		noalias(outputDelta) = trans(coefficients);

		computeDelta(delta,state,true);
		inputDerivative.resize(numPatterns,inputSize());
		noalias(inputDerivative) = trans(rows(delta,0,inputSize()));
	}
	
	virtual void weightedDerivatives(
		BatchInputType const & patterns,
		BatchOutputType const & coefficients,
		State const& state,
		RealVector& parameterDerivative,
		BatchInputType& inputDerivative
	)const{
		SIZE_CHECK(coefficients.size2() == m_outputNeurons);
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		std::size_t numPatterns = patterns.size1();
		
		
		//compute full delta and thus the input derivative
		RealMatrix delta(numberOfNeurons(),numPatterns,0.0);
		RealSubMatrix outputDelta = rows(delta,delta.size1()-outputSize(),delta.size1());
		noalias(outputDelta) = trans(coefficients);
		
		computeDelta(delta,state,true);
		inputDerivative.resize(numPatterns,inputSize());
		noalias(inputDerivative) = trans(rows(delta,0,inputSize()));
		
		//reuse delta to compute the parameter derivative
		computeParameterDerivative(delta,state,parameterDerivative);
	}
	
	//! \brief Calculates the derivative for the special case, when error terms for all neurons of the network exist.
	//!
	//! This is usefull when the hidden neurons need to meet additional requirements.
	//! The value of delta is changed during computation and holds the results of the backpropagation steps.
	//! The format is such that the rows of delta are the neurons and the columns the patterns.
	void weightedParameterDerivativeFullDelta(
		RealMatrix const& patterns, RealMatrix& delta, State const& state, RealVector& gradient
	)const{
		InternalState const& s = state.toState<InternalState>();
		SIZE_CHECK(delta.size1() == m_numberOfNeurons);
		SIZE_CHECK(delta.size2() == patterns.size1());
		SIZE_CHECK(s.responses.size2() == patterns.size1());
		
		computeDelta(delta,state,false);
		//now compute the parameter derivative from the delta values
		computeParameterDerivative(delta,state,gradient);
	}

	//!  \brief Creates a connection matrix for a network.
	//!
	//! Automatically creates a network with several layers, with
	//! the numbers of neurons for each layer defined by \em layers.
	//! layers must be at least size 2, which will result in a network with no hidden layers.
	//! the first and last values correspond to the number of inputs and outputs respectively.
	//!
	//! The network supports three different tpes of connection models:
	//! FFNetStructures::Normal corresponds to a layerwise connection between consecutive
	//! layers. FFNetStructures::InputOutputShortcut additionally adds a shortcut between
	//! input and output neurons. FFNetStructures::Full connects every layer to every following
	//! layer, this includes also the shortcuts for input and output neurons. Additionally
	//! a bias term an be used.
	//!
	//! While Normal and Full only use the layer matrices, inputOutputShortcut also uses
	//! the corresponding matrix variable (be aware that in the case of only one hidden layer,
	//! the shortcut between input and output leads to the same network as the Full - in that case
	//! the Full topology is chosen for optimization reasons)
	//!
	//! \param  layers contains the numbers of neurons for each layer of the network.
	//! \param  connectivity type of connection used between layers
	//! \param  bias       if set to \em true, connections from
	//!                    all neurons (except the input neurons)
	//!                    to the bias will be set.
	void setStructure(
		std::vector<size_t> const& layers,
		FFNetStructures::ConnectionType connectivity = FFNetStructures::Normal,
		bool biasNeuron = true
	){
		SIZE_CHECK(layers.size() >= 2);
		m_layerMatrix.resize(layers.size()-1);//we don't model the input layer
		m_backpropMatrix.resize(layers.size()-1);//we don't model the output layer
		
		//small optimization for ntworks with only 3 layers
		//in this case, we don't need an explicit shortcut as we can integrate it into
		//the big matrices
		if(connectivity == FFNetStructures::InputOutputShortcut && layers.size() ==3)
			connectivity = FFNetStructures::Full;
		
		
		m_inputNeurons = layers.front();
		m_outputNeurons = layers.back();
		m_numberOfNeurons = 0;
		for(std::size_t i = 0; i != layers.size(); ++i){
			m_numberOfNeurons += layers[i];
		}
		if(biasNeuron){
			m_bias.resize(m_numberOfNeurons - m_inputNeurons);
		}
		
		if(connectivity == FFNetStructures::Full){
			//connect to all previous layers.
			std::size_t numNeurons = layers[0];
			for(std::size_t i = 0; i != m_layerMatrix.size(); ++i){
				m_layerMatrix[i].resize(layers[i+1],numNeurons);
				m_backpropMatrix[i].resize(layers[i],m_numberOfNeurons-numNeurons);
				numNeurons += layers[i+1];
				
			}
			m_inputOutputShortcut.resize(0,0);
		}else{
			//only connect with the previous layer
			for(std::size_t i = 0; i != m_layerMatrix.size(); ++i){
				m_layerMatrix[i].resize(layers[i+1],layers[i]);
				m_backpropMatrix[i].resize(layers[i],layers[i+1]);
			}
			
			//create a shortcut from input to output when desired
			if(connectivity == FFNetStructures::InputOutputShortcut){
				m_inputOutputShortcut.resize(m_outputNeurons,m_inputNeurons);
			}
		}
	}
	
	//!  \brief Creates a connection matrix for a network with a
	//!         single hidden layer
	//!
	//!  Automatically creates a network with
	//!  three different layers: An input layer with \em in input neurons,
	//!  an output layer with \em out output neurons and one hidden layer
	//!  with \em hidden neurons, respectively.
	//!
	//!  \param  in         number of input neurons.
	//!  \param  hidden    number of neurons of the second hidden layer.
	//!  \param  out        number of output neurons.
	//!  \param  connectivity  Type of connectivity between the layers
	//!  \param  bias       if set to \em true, connections from
	//!                     all neurons (except the input neurons)
	//!                     to the bias will be set.
	void setStructure(
		std::size_t in,
		std::size_t hidden,
		std::size_t out,
		FFNetStructures::ConnectionType connectivity = FFNetStructures::Normal,
		bool bias      = true
	){
		std::vector<size_t> layer(3);
		layer[0] = in;
		layer[1] = hidden;
		layer[2] = out;
		setStructure(layer, connectivity, bias);
	}
	
	//!  \brief Creates a connection matrix for a network with two
	//!         hidden layers.
	//!
	//!  Automatically creates a network with
	//!  four different layers: An input layer with \em in input neurons,
	//!  an output layer with \em out output neurons and two hidden layers
	//!  with \em hidden1 and \em hidden2 hidden neurons, respectively.
	//!
	//!  \param  in         number of input neurons.
	//!  \param  hidden1    number of neurons of the first hidden layer.
	//!  \param  hidden2    number of neurons of the second hidden layer.
	//!  \param  out        number of output neurons.
	//!  \param  connectivity  Type of connectivity between the layers
	//!  \param  bias       if set to \em true, connections from
	//!                     all neurons (except the input neurons)
	//!                     to the bias will be set.
	void setStructure(
		std::size_t in,
		std::size_t hidden1,
		std::size_t hidden2,
		std::size_t out,
		FFNetStructures::ConnectionType connectivity = FFNetStructures::Normal,
		bool bias      = true
	){
		std::vector<size_t> layer(4);
		layer[0] = in;
		layer[1] = hidden1;
		layer[2] = hidden2;
		layer[3] = out;
		setStructure(layer, connectivity, bias);
	}

	//! \brief Configures the network.
	//!
	//!  The Data format is as follows:
	//!   general properties:
	//!  "inputs" number of input neurons. No default value, must be set!
	//!  "outputs" number of output neurons. No default value, must be set!
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
		std::vector<size_t> layers(numLayers+2);
		if(numLayers){
			layers[0]=inputNeurons;
			layers.back()=outputNeurons;
			for(Iter layer=range.first;layer!=range.second;++layer){
				size_t layerPos = layer->second.get<size_t>("number");
				size_t layerSize = layer->second.get<size_t>("neurons");
				if(layerPos > numLayers)
					SHARKEXCEPTION("[FFNet::configure] layer number too big");
				layers[layerPos+1] = layerSize;
			}
			
		}
		if(allConnections){
			setStructure(layers,FFNetStructures::Full,biasConnections);
		}
		else if(inOutConnections){
			setStructure(layers,FFNetStructures::InputOutputShortcut,biasConnections);
		}else{
			setStructure(layers,FFNetStructures::Normal,biasConnections);
		}
	}

	//! From ISerializable, reads a model from an archive
	void read( InArchive & archive ){
		archive>>m_inputNeurons;
		archive>>m_outputNeurons;
		archive>>m_numberOfNeurons;
		archive>>m_layerMatrix;
		archive>>m_backpropMatrix;
		archive>>m_inputOutputShortcut;
		archive>>m_bias;
	}

	//! From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const{
		archive<<m_inputNeurons;
		archive<<m_outputNeurons;
		archive<<m_numberOfNeurons;
		archive<<m_layerMatrix;
		archive<<m_backpropMatrix;
		archive<<m_inputOutputShortcut;
		archive<<m_bias;
	}


private:
	
	void computeDelta(
		RealMatrix& delta, State const& state, bool computeInputDelta
	)const{
		SIZE_CHECK(delta.size1() == numberOfNeurons());
		InternalState const& s = state.toState<InternalState>();

		//initialize output neurons using coefficients
		RealSubMatrix outputDelta = rows(delta,delta.size1()-outputSize(),delta.size1());
		ConstRealSubMatrix outputResponse = rows(s.responses,delta.size1()-outputSize(),delta.size1());
		noalias(outputDelta) *= m_outputNeuron.derivative(outputResponse);

		//iterate backwards using the backprop matrix and propagate the errors to get the needed delta values
		//we stop until we have filled all delta values. Thus we might not necessarily compute all layers.
		
		//last neuron of the current layer that we need to compute
		//we don't need (or can not) compute the values of the output neurons as they are given from the outside
		std::size_t endNeuron = delta.size1()-outputSize();
		std::size_t layer = m_backpropMatrix.size()-1;
		std::size_t endIndex = computeInputDelta? 0: inputSize();
		while(endNeuron > endIndex){
			
			RealMatrix const& weights = m_backpropMatrix[layer];
			std::size_t beginNeuron = endNeuron - weights.size1();//first neuron of the current layer
			//get the delta and response values of this layer
			RealSubMatrix layerDelta = rows(delta,beginNeuron,endNeuron);
			RealSubMatrix layerDeltaInput = rows(delta,endNeuron,endNeuron+weights.size2());
			ConstRealSubMatrix layerResponse = rows(s.responses,beginNeuron,endNeuron);

			axpy_prod(weights,layerDeltaInput,layerDelta,false);//add the values to the maybe non-empty delta part
			if(layer != 0){
				noalias(layerDelta) *= m_hiddenNeuron.derivative(layerResponse);
			}
			//go a layer backwards
			endNeuron=beginNeuron;
			--layer;
		}
		
		//add the shortcut deltas if necessary
		if(inputOutputShortcut().size1() != 0)
			axpy_prod(trans(inputOutputShortcut()),outputDelta,rows(delta,0,inputSize()),false);
	}
	
	void computeParameterDerivative(RealMatrix const& delta, State const& state, RealVector& gradient)const{
		SIZE_CHECK(delta.size1() == numberOfNeurons());
		InternalState const& s = state.toState<InternalState>();
		// calculate error gradient
		//todo: take network structure into account to prevent checking all possible weights...
		gradient.resize(numberOfParameters());
		std::size_t pos = 0;
		std::size_t layerStart = inputSize();
		for(std::size_t layer = 0; layer != layerMatrices().size(); ++layer){
			std::size_t layerRows =  layerMatrices()[layer].size1();
			std::size_t layerColumns =  layerMatrices()[layer].size2();
			std::size_t params = layerRows*layerColumns;
			axpy_prod(
				rows(delta,layerStart,layerStart+layerRows),
				trans(rows(s.responses,layerStart-layerColumns,layerStart)),
				//interpret part of the gradient as the weights of the layer
				to_matrix(subrange(gradient,pos,pos+params),layerRows,layerColumns)
			);
			pos += params;
			layerStart += layerRows;
		}
		//check whether we need the bias derivative
		if(!bias().empty()){
			//calculate bias derivative
			for (std::size_t neuron = m_inputNeurons; neuron < m_numberOfNeurons; neuron++){
				gradient(pos) = sum(row(delta,neuron));
				pos++;
			}
		}
		//compute shortcut derivative
		if(inputOutputShortcut().size1() != 0){
			std::size_t params = inputSize()*outputSize();
			axpy_prod(
				rows(delta,delta.size1()-outputSize(),delta.size1()),
				trans(rows(s.responses,0,inputSize())),
				to_matrix(subrange(gradient,pos,pos+params),outputSize(),inputSize())
			);
		}
		
	}
	

	//! \brief Number of all network neurons.
	//!
	//! This is the total number of neurons in the network, i.e.
	//! input, hidden and output neurons.
	std::size_t m_numberOfNeurons;
	std::size_t m_inputNeurons;
	std::size_t m_outputNeurons;

	//! \brief represents the connection matrix using a layered structure for forward propagation
	//!
	//! a layer is made of neurons with consecutive indizes which are not
	//! connected with each other. In other words, if there exists a k i<k<j such
	//! that C(i,k) = 1 or C(k,j) = 1 or C(j,i) = 1 than the neurons i,j are not in the same layer.
	//! This is the forward view, meaning that the layers holds the weights which are used to calculate
	//! the activation of the neurons of the layer.
	std::vector<RealMatrix> m_layerMatrix;
	
	//! \brief optional matrix directly connecting input to output
	//!
	//! This is only filled when the ntworkhas an input-output shortcut but not a full layer connection.
	RealMatrix m_inputOutputShortcut;
	
	//!\brief represents the backwards view of the network as layered structure.
	//!
	//! This is the backward view of the Network which is used for the backpropagation step. So every
	//! Matrix contains the weights of the neurons which are activatived by the layer.
	std::vector<RealMatrix> m_backpropMatrix;

	//! bias weights of the neurons
	RealVector m_bias;

	//!Type of hidden neuron. See Models/Neurons.h for a few choices
	HiddenNeuron m_hiddenNeuron;
	//! Type of output neuron. See Models/Neurons.h for a few choices
	OutputNeuron m_outputNeuron;
};


}
#endif