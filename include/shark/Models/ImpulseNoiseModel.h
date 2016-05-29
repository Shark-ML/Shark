/*!
 * 
 *
 * \brief       Implements a Model which corrupts the input by setting values of the input to a given value
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
#ifndef SHARK_MODELS_ImpulseNOISEMODEL_H
#define SHARK_MODELS_ImpulseNOISEMODEL_H

#include <shark/Models/AbstractModel.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Core/OpenMP.h>
namespace shark {

/// \brief Model which corrupts the data using Impulse noise
///
/// We define impulse noise as a noise which randomly sets the value of an element
/// in a vector to a given value, 0 by default. We chose the name as with for example
/// a noise of 1, impulses can be seen in the visualised vectors. 
///
/// very input dimension is tested independently.
///
/// This noise can be used to implement denoising autoencoders for binary data.
class ImpulseNoiseModel : public AbstractModel<RealVector,RealVector>
{
private:
	std::size_t m_numInputs;
	double m_prob;
	double m_value;
public:


	/// Default Constructor; use setStructure later
	ImpulseNoiseModel(){
		m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
	}
	/// Constructor creating a model with given input size and a probability to set values of the input to a given value.
	ImpulseNoiseModel(unsigned int inputs, double prob, double value = 0.0)
	: m_numInputs(inputs), m_prob(prob), m_value(value){
		m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ImpulseNoiseModel"; }

	/// obtain the input dimension
	size_t inputSize() const{
		return m_numInputs;
	}

	/// obtain the output dimension
	size_t outputSize() const{
		return m_numInputs;
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
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	/// \brief Add noise to the input
	void eval(BatchInputType const& inputs, BatchOutputType& outputs)const{
		SIZE_CHECK(inputs.size2() == inputSize());
		SHARK_CRITICAL_REGION{
			outputs = inputs;
			for(std::size_t i = 0; i != outputs.size1(); ++i){
				for(std::size_t j = 0; j != outputs.size2(); ++j){
					if(Rng::coinToss(m_prob)){
						outputs(i,j) = m_value;
					}
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
