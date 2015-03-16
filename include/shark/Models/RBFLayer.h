/*!
 * 
 *
 * \brief      Implements a radial basis function layer.
 * 
 * 
 *
 * \author      O. Krause
 * \date        2014
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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
#ifndef SHARK_MODELS_RBFLayer_H
#define SHARK_MODELS_RBFLayer_H

#include <shark/Models/AbstractModel.h>
#include <boost/math/constants/constants.hpp>
namespace shark {

///  \brief Implements a layer of radial basis functions in a neural network.
/// 
/// A Radial basis function layer as modeled in shark is a set of N
/// Gaussian distributions \f$ p(x|i) \f$.
/// \f[
///   p(x|i) = e^{\gamma_i*\|x-m_i\|^2}
/// \f]
/// and the layer transforms an input x to a vector \f$(p(x|1),\dots,p(x|N)\f$.
///  The \f$\gamma_i\f$ govern the width of the Gaussians, while the
///  vectors \f$ m_i \f$ set the centers of every Gaussian distribution. 
///
/// RBF networks profit much from good guesses on the centers and
/// kernel function parameters.  In case of a Gaussian kernel a call
/// to k-Means or the EM-algorithm can be used to get a good
/// initialisation for the network.
class RBFLayer : public AbstractModel<RealVector,RealVector>
{
private:
	struct InternalState: public State{
		RealMatrix norm2;
		RealMatrix p;
		
		void resize(std::size_t numPatterns, std::size_t numNeurons){
			norm2.resize(numPatterns,numNeurons);
			p.resize(numPatterns,numNeurons);
		}
	};

public:
	///  \brief Creates an empty Radial Basis Function layer.
	RBFLayer();
	
	///  \brief Creates a layer of a Radial Basis Function Network.
	///
	///  This method creates a Radial Basis Function Network (RBFN) with
	///  \em numInput input neurons and \em numOutput output neurons.
	///
	///  \param  numInput  Number of input neurons, equal to dimensionality of
	///                    input space.
	///  \param  numOutput Number of output neurons, equal to dimensionality of
	///                    output space and number of gaussian distributions
	RBFLayer(std::size_t numInput, std::size_t numOutput);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RBFLayer"; }

	///\brief Returns the current parameter vector. The amount and order of weights depend on the training parameters.
	///
	///The format of the parameter vector is \f$ (m_1,\dots,m_k,\log(\gamma_1),\dots,\log(\gamma_k))\f$
	///if training of one or more parameters is deactivated, they are removed from the parameter vector
	RealVector parameterVector()const;
	
	///\brief Sets the new internal parameters.
	void setParameterVector(RealVector const& newParameters);
	
	///\brief Returns the number of parameters which are currently enabled for training.
	std::size_t numberOfParameters()const;

	///\brief Returns the number of input neurons.
	std::size_t inputSize()const{
		return m_centers.size2();
	}
	
	///\brief Returns the number of output neurons.
	std::size_t outputSize()const{
		return m_centers.size1();
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}
	
	
	///  \brief Configures a Radial Basis Function Network.
	///
	///  This method initializes the structure of the Radial Basis Function Network (RBFN) with
	///  \em numInput input neurons, \em numOutput output neurons and \em numHidden
	///  hidden neurons.
	///
	///  \param  numInput  Number of input neurons, equal to dimensionality of
	///                    input space.
	///  \param  numOutput Number of output neurons (basis functions), equal to dimensionality of
	///                    output space.
	void setStructure(std::size_t numInput, std::size_t numOutput);

	
	using AbstractModel<RealVector,RealVector>::eval;
	void eval(BatchInputType const& patterns, BatchOutputType& outputs, State& state)const;
	

	void weightedParameterDerivative(
		BatchInputType const& pattern, BatchOutputType const& coefficients, State const& state, RealVector& gradient
	)const;

	///\brief Enables or disables parameters for learning.
	///
	/// \param centers whether the centers should be trained
	/// \param width whether the distribution width should be trained
	void setTrainingParameters(bool centers, bool width);

	///\brief Returns the center values of the neurons.
	BatchInputType const& centers()const{
		return m_centers;
	}
	///\brief Sets the center values of the neurons.
	BatchInputType& centers(){
		return m_centers;
	}
	
	///\brief Returns the width parameter of the Gaussian functions 
	RealVector const& gamma()const{
		return m_gamma;
	}
	
	/// \brief sets the width parameters - the gamma values - of the distributions.
	void setGamma(RealVector const& gamma);
	
	/// From ISerializable, reads a model from an archive
	void read( InArchive & archive );

	/// From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const;
protected:
	//====model parameters

	///\brief The center points. The i-th element corresponds to the center of neuron number i
	RealMatrix m_centers;
	
	///\brief stores the width parameters of the Gaussian functions
	RealVector m_gamma;

	/// \brief the logarithm of the normalization constant for every distribution
	RealVector m_logNormalization;

	//=====training parameters
	///enables learning of the center points of the neurons
	bool m_trainCenters;
	///enables learning of the width parameters.
	bool m_trainWidth;



};
}

#endif

