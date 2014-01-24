/*!
 * 
 * \file        RBFNet.h
 *
 * \brief       Offers the functions to create and to work with a
 * radial basis fucntion network
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010
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
#ifndef SHARK_MODELS_RBFNET_H
#define SHARK_MODELS_RBFNET_H

#include <shark/Models/AbstractModel.h>
#include <shark/Core/SharedVector.h>
namespace shark {

///  \brief Offers the functions to create and to work with radial basis function networks
/// 
/// A Radial basis function network as modeled in shark is a linear 
/// combination of unnormalized Gaussian distributions \f$ p_j(x) \f$.
/// \f[
///   p_j(x) = e^{\gamma_j*\|x-c_\j|^2}
/// \f]
///  The \f$\gamma_j\f$ govern the width of the Gaussian, while the
///  vectors $c_j$ set the centers of every Gaussian neuron.  After
///  calculating the reponses of the neurons to the input, the result
///  \f$y\f$ is then calculated using an affine linear function using
///  the weightmatrix \f$W\f$ as well as an bias term \f$b\f$
/// \f[
///   y(x) = b + Wp(x)
/// \f]
///
/// RBF networks profit much from good guesses on the centers and
/// kernel function parameters.  In case of a Gaussian kernel a call
/// to k-Means or the EM-algorithm can be used to get a good
/// initialisation for the network.
class RBFNet : public AbstractModel<RealVector,RealVector>
{
private:
	struct InternalState: public State{
		RealMatrix norm2;
		RealMatrix expNorm;
		
		void resize(std::size_t numPatterns, std::size_t numNeurons){
			norm2.resize(numPatterns,numNeurons);
			expNorm.resize(numPatterns,numNeurons);
		}
	};
	
	void computeGaussianResponses(BatchInputType const& patterns, InternalState& state)const;

public:
	///  \brief Creates an empty Radial Basis Function Network. A call to configure is required afterwards.
	RBFNet();
	
	///  \brief Creates a Radial Basis Function Network.
	///
	///  This method creates a Radial Basis Function Network (RBFN) with
	///  \em numInput input neurons, \em numOutput output neurons and \em numHidden
	///  hidden neurons.
	///
	///  \param  numInput  Number of input neurons, equal to dimensionality of
	///                    input space.
	///  \param  numOutput Number of output neurons, equal to dimensionality of
	///                    output space.
	///  \param  numHidden Number of hidden neurons.
	RBFNet(std::size_t numInput, std::size_t numHidden, std::size_t numOutput);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RBFNet"; }

	///\brief Returns the current parameter vector. The amount and order of weights depend on the training parameters.
	///
	///This may be a slow operation, since the vector is everytime constructed from scratch
	///So don't call it needlessly.
	///the format of the parameter vector is \f$ (W,b,m_1,\dots,m_k,\log(\gamma_1),\dots,\log(\gamma_k))\f$
	///if training of one or more parameters is deactivated, they are removed from the parameter vector
	RealVector parameterVector()const;
	
	///\brief Sets the new internal parameters.
	void setParameterVector(RealVector const& newParameters);

	///\brief Returns the number of input neurons.
	std::size_t inputSize()const{
		return m_inputNeurons;
	}
	
	///\brief Returns the number of output neurons.
	std::size_t outputSize()const{
		return m_outputNeurons;
	}
	
	///\brief Returns the number of hidden neurons.
	std::size_t numHiddens()const{
		return m_centers.size1();
	}

	///\brief Returns the number of parameters which are currently enabled for training.
	///
	///at every call of this method, the number is calculated from scratch!
	std::size_t numberOfParameters()const;
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}


	///\brief Configures the RBFNet using a property tree.
	///
	/// the following properties are needed:
	/// "inputs" number of input neurons.
	/// "outputs" number of output neurons.
	/// "hidden" number of hidden neurons.
	/// the following values are enabled as default
	/// "trainLinear"  should the linear output be trained?
	/// "trainCenters" should the centers be trained?
	/// "trainWidth" should the widght of the distributions be trained?
	/// deactivation one of these parameters will remove them from #parameterVector
	/// optional properties:
	/// This is particularly useful together with trainKernels = false
	/// See the documentation of the desired Kernel fr further details about the contents of this node.
	void configure( PropertyTree const& node );
	
	
	///  \brief Configures a Radial Basis Function Network.
	///
	///  This method initializes the structure of the Radial Basis Function Network (RBFN) with
	///  \em numInput input neurons, \em numOutput output neurons and \em numHidden
	///  hidden neurons.
	///
	///  \param  numInput  Number of input neurons, equal to dimensionality of
	///                    input space.
	///  \param  numOutput Number of output neurons, equal to dimensionality of
	///                    output space.
	///  \param  numHidden Number of hidden neurons.
	void setStructure(std::size_t numInput, std::size_t numHidden, std::size_t numOutput);


	void eval(BatchInputType const& patterns, BatchOutputType& outputs, State& state)const;
	using AbstractModel<RealVector,RealVector>::eval;

	void weightedParameterDerivative(
		BatchInputType const& pattern, BatchOutputType const& coefficients, State const& state, RealVector& gradient
	)const;

	///\brief Enables or disables parameters for learning.
	///
	/// \param linear whether the linear output weights sho9uld be trained
	/// \param centers whether the centers should be trained
	/// \param width whether the distribution width should be trained
	void setTrainingParameters(bool linear,bool centers, bool width);

	///\brief Returns the center values of the neurons.
	BatchInputType const& centers()const{
		return m_centers;
	}
	///\brief Sets the center values of the neurons.
	void setCenter(std::size_t i, RealVector const& center){
		noalias(row(m_centers,i)) = center;
	}
	///\brief Returns the linear weights of the output neurons.
	RealMatrix const& linearWeights()const{
		return m_linearWeights;
	}
	///\brief Returns the linear weights of the output neurons.
	void setlinearWeights(RealMatrix const& linearWeights){
		m_linearWeights = linearWeights;
	}
	///\brief Returns the bias of the output neurons.
	RealVector const& bias()const{
		return m_bias;
	}
	///\brief Returns the bias of the output neurons.
	void setBias(RealVector const& bias){
		m_bias = bias;
	}
	
	///\brief Returns the width parameter of the Gaussian functions 
	RealVector const& gamma()const{
		return m_gamma;
	}
	
	void setGamma(RealVector const& gamma){
		m_gamma = gamma;
	}
	
	/// From ISerializable, reads a model from an archive
	void read( InArchive & archive );

	/// From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const;
protected:
	///the size of the input vector the network expects
	std::size_t m_inputNeurons;
	///the size of the output vector the network produces
	std::size_t m_outputNeurons;

	//====model parameters

	///\brief The center points. The i-th element corresponds to the center of neuron number i
	RealMatrix m_centers;
	///\brief Weights of the linear part of the network. m_linearWeights(i,j) connects output neuron i with hidden neuron j
	RealMatrix m_linearWeights;
	///\brief Bias values of the output layer. m_bias(i) is the bias of output neuron i
	RealVector m_bias;
	
	///\brief stores the width parameters of the Gaussian functions
	RealVector m_gamma;

	//=====training parameters
	///enables learning of linear Weights and the bias Neurons
	bool m_trainLinear;
	///enables learning of the center points of the neurons
	bool m_trainCenters;
	///enables learning of the width parameters.
	bool m_trainWidth;



};
}

#endif

