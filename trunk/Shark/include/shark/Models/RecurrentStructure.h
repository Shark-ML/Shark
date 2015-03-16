//===========================================================================
 //!  \brief Offers a basic structure for recurrent networks
 //!
 //!  \author  O. Krause
 //!  \date    2011
 //!
 //!  \par Copyright (c) 1999-2001:
 //!      Institut f&uuml;r Neuroinformatik<BR>
 //!      Ruhr-Universit&auml;t Bochum<BR>
 //!      D-44780 Bochum, Germany<BR>
 //!      Phone: +49-234-32-27974<BR>
 //!      Fax:   +49-234-32-14209<BR>
 //!      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 //!      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 //!
 //!
 //!
 //!  <BR><HR>
 //!  This file is part of Shark. This library is free software;
 //!  you can redistribute it and/or modify it under the terms of the
 //!  GNU General Public License as published by the Free Software
 //!  Foundation; either version 3, or (at your option) any later version.
 //!
 //!  This library is distributed in the hope that it will be useful,
 //!  but WITHOUT ANY WARRANTY; without even the implied warranty of
 //!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 //!  GNU General Public License for more details.
 //!
 //!  You should have received a copy of the GNU General Public License
 //!  along with this library; if not, see <http://www.gnu.org/licenses/>.
#ifndef SHARK_ML_MODEL_RECURENTNETWORK_H
#define SHARK_ML_MODEL_RECURENTNETWORK_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/ISerializable.h>
namespace shark{
//!  \brief Offers a basic structure for recurrent networks.
//!
//! it is possible to define the tzpe of sigmoids and the form of the connection matrix.
//! this structure can be shared between different types of ents like the RNNet and the OnlineRNNet
class RecurrentStructure: public ISerializable
{
public:
	//! Creates an empty recurrent neural network.
	//! A call to setStructure is needed afterwards to configure the topology of the network
	RecurrentStructure();


	//! type enum for the different variants of sigmoids
	enum SigmoidType{
		///f(x) = x
		Linear,
		///f(x) = 1/(1+exp(-x))
		Logistic,
		///f(x) = tanh(x)
		Tanh,
		/// f(x) = x/(1+|x|)
		FastSigmoid 
	};

	//!returns the connection Matrix of the network.
	//!
	//!The format is best described with an example:
	//!given a Network with 2 inputs, 2 outputs and 2 hidden and no bias unit,
	//!where the inputs are only connected to the hidden units.
	//!The corresponding matrix looks like this:
	//!
	//!1 2 3 4 5 6 7
	//!1 1 0 1 1 1 1 first hidden
	//!1 1 0 1 1 1 1 second hidden
	//!0 0 0 1 1 1 1 first output
	//!0 0 0 1 1 1 1 second output
	//!
	//!The ith row stands for the ith neuron of the network and when an element
	//!(i,j) is 1, the ith unit will receive input from unit j.
	//! the first =0,..,inputs-1 columns are the input neurons followd by the column of the bias, 
	//! which is completely zero in this example
	//!if j is a hidden or output neuron, the activation from the PREVIOUS time step
	//!is used. if j is an input neuron, the current input is used.
	//!input neurons can't receive activation. This is no limitation, since the hidden
	//!layer can be subdivided in arbitrary sublayers when the right topology is used.
	//! the last column of the matrix is reserved for the bias neuron. So the matrix has size
	//! NxN+1
	const IntMatrix& connections()const{
		return m_connectionMatrix;
	}
	//! returns whether the connection between neuron i and j exists
	bool connection(std::size_t i, std::size_t j)const{
		return m_connectionMatrix(i,j);
	}

	//!returns the current weight matrix
	const RealMatrix& weights()const{
		return m_weights;
	}
	//! returns the weight of the connection between neuron i and j
	double weight(std::size_t i, std::size_t j)const{
		return m_weights(i,j);
	}

	//!returns the type of sigmoid used in this network
	SigmoidType sigmoidType() const {
		return m_sigmoidType;
	}

	//!sets the type of sigmoid used in this network
	//!\param sigmoidType the type of sigmoid
	void setSigmoidType(SigmoidType sigmoidType){
		m_sigmoidType = sigmoidType;
	}

	//!Sets the weight matrix. It is not allowed that elements are non-zero
	//!when the element in the connection matrix is 0!
	//!\param weights the new weight matrix
	void setWeights(const RealMatrix& weights);

	//!  \brief Based on a given connection matrix a network is created.
	//!
	//!  This method needs to know how many inputs and outputs the network has
	//!  and how the units are connected.
	//!
	//!  If a standard structure is needed, see the other version of this method.
	//!  Also see #connections for a quick explanation of the matrix format
	//!
	//!  The same mechanic applies alo to recurrentConnections
	//!  but every element can be set, not only the lower triangular part.
	//!
	//! After this operation, all weights are initialized to 0.
	//!
	//!
	//! \param inputs number of input neurons of the network
	//! \param outputs number of output neurons of the network
	//! \param connections feed-forward connections. default is true
	//! \param sigmoidType the type of the sigmoid to be used. the default is the Logistic function
	void setStructure(std::size_t inputs, std::size_t outputs, const IntMatrix& connections, SigmoidType sigmoidType = Logistic);


	//! \brief Creates a fully connected topology for the network with optional bias
	//!
	//!  After a call, the network will have hidden+out units.
	//!
	//!
	//! \param in number of input neurons
	//! \param hidden number of output neurons
	//! \param out number of input neurons
	//! \param bias enables bias neuron, default is true
	//! \param sigmoidType the type of the sigmoid to be used. the default is the Logistic function
	void setStructure(std::size_t in, std::size_t hidden, std::size_t out, bool bias = true, SigmoidType sigmoidType = Logistic);

	//! get internal parameters of the model
	RealVector parameterVector() const;
	
	//! set internal parameters of the model
	void setParameterVector(RealVector const& newParameters);

	//! From ISerializable, reads the Network from an archive
	void read( InArchive & archive );

	//! From ISerializable, writes the Network to an archive
	void write( OutArchive & archive ) const;

	//! The number of input neurons of the network
	std::size_t inputs()const{
		return m_inputNeurons;
	}
	//! The number of output neurons of the network
	std::size_t outputs()const{
		return m_outputNeurons;
	}
	std::size_t numberOfNeurons()const{
		return m_numberOfNeurons;
	}
	std::size_t numberOfUnits()const{
		return m_numberOfUnits;
	}

	//! The index of the bias unit
	std::size_t bias()const{
		return m_bias;
	}

	//! number of parameters of the network
	std::size_t parameters()const{
		return m_numberOfParameters;
	}

	//! Activation function for a neuron.
	double neuron(double activation);

	//! Computes the derivative of the neuron.
	double neuronDerivative(double activation);

protected:

	//================Convenience index variables=====================
	//! The total number of neurons of the network (input, output and hidden).
	std::size_t m_numberOfNeurons;

	//! total number units of the network (input, output, hidden and bias)
	std::size_t m_numberOfUnits;

	//! The number of input neurons of the network
	std::size_t m_inputNeurons;
	//! The number of output neurons of the network
	std::size_t m_outputNeurons;
	//! The number of hidden neurons of the network
	std::size_t m_hidden;

	//! index of the bias unit
	std::size_t m_bias;

	//! type of Sigmoid used by the network
	SigmoidType m_sigmoidType;

	//===================network variables========================
	//! The absolute number of parameters of the network
	std::size_t m_numberOfParameters;

	//! stores the topology of the network.
	//! Element (i,j) is 1 if the ith neuron receives input from neuron j.
	//! The data for neuron i is stored in the ith row of the matrix.
	IntMatrix m_connectionMatrix;

	//! stores the feed-forward part of the weights. the recurrent part is added
	//! via m_recurrentWeights. The weights for neuron i are stored in the ith row of the matrix
	RealMatrix m_weights;
};
}

#endif //RNNET_H









