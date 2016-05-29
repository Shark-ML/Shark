/*!
 * 
 *
 * \brief       Implements a Model using a linear function.
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
#ifndef SHARK_MODELS_GAUSSIANNOISEMODEL_H
#define SHARK_MODELS_GAUSSIANNOISEMODEL_H

#include <shark/Models/AbstractModel.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Core/OpenMP.h>
namespace shark {

/// \brief Model which corrupts the data using gaussian noise
///
/// When training autoencoders, it proved beneficial to add noise to the input
/// and train the model to remove that noise again, instead of only larning a
/// identity transformation. This Model represents one choice of Noise: Gaussian Noise,
/// to do this. the formula of corruption of an input \f$ x=(x_1,\dots,x_n) \f$ with variances
/// \f$ \sigma = (\sigma_1, \dots, \sigma_n) \f$ is 
/// \f[ x_i \leftarrow x_i + N(0,\sigma_i) \f]
///
/// Usage is simple. given your autoencoder/decoder pair
///   ConvatenatedModel<RealVector,RealVector> autoencoder = encoder >> decoder;
/// we can just concatenate this model:
///   GaussianNoiseModel noise(0.1);//variance of noise
///   ConvatenatedModel<RealVector,RealVector> denoisingAutoencoder = noise>>autoencoder;
/// and train the model using the standard autoencoder error
class GaussianNoiseModel : public AbstractModel<RealVector,RealVector>
{
private:
	RealVector m_variances;
public:


	/// Default Constructor; use setStructure later
	GaussianNoiseModel(){
		m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
	}
	/// Constructor creating a model with given input size and the same variance for all inputs
	GaussianNoiseModel(unsigned int inputs, double variance)
	: m_variances(inputs,variance){
		m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "GaussianNoiseModel"; }

	/// obtain the input dimension
	size_t inputSize() const{
		return m_variances.size();
	}

	/// obtain the output dimension
	size_t outputSize() const{
		return m_variances.size();
	}

	/// obtain the parameter vector
	RealVector parameterVector() const{
		return RealVector();
	}

	/// overwrite the parameter vector
	void setParameterVector(RealVector const& newParameters)
	{
		SIZE_CHECK(newParameters.size() == 0);
	}

	/// return the number of parameter
	size_t numberOfParameters() const{
		return 0;
	}

	/// overwrite structure and parameters
	void setStructure(unsigned int inputs, double variance){
		m_variances = RealVector(inputs,variance);
	}

	/// overwrite structure and parameters
	void setStructure(RealVector const& variances){
		m_variances = variances;
	}
	
	RealVector const& variances() const{
		return m_variances;
	}
	
	RealVector& variances(){
		return m_variances;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	/// \brief Add noise to the input
	void eval(BatchInputType const& inputs, BatchOutputType& outputs)const{
		SIZE_CHECK(inputs.size2() == inputSize());
		//we use the global Rng here so if this is a threaded region, we might
		//run into troubles when multiple threads run this. This should not be a bottle neck
		//as this routine should be quite fast, while very expensive routines are likely to
		//follow in the networks following this.
		SHARK_CRITICAL_REGION{
			outputs = inputs;
			for(std::size_t i = 0; i != outputs.size1(); ++i){
				for(std::size_t j = 0; j != outputs.size2(); ++j){
					outputs(i,j) += Rng::gauss(0,m_variances(j));
				}
			}
		}
	}
	/// Evaluate the model: output = matrix * input + offset
	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State& state)const{
		eval(inputs,outputs);
	}
	
	void weightedParameterDerivative(
		BatchInputType const& patterns, RealVector const& coefficients, State const& state, RealVector& gradient
	)const{
		gradient.resize(0);
	}
};


}
#endif
