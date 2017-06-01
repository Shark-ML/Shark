/*!
 * 
 *
 * \brief       -
 *
 * \author      O.Krause
 * \date        2011
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
#ifndef MODELS_NEURONS_H
#define MODELS_NEURONS_H
 
#include <shark/LinAlg/Base.h>
#include <shark/Models/AbstractModel.h>
 
namespace shark{
	
///\brief Neuron which computes the hyperbolic tangenst with range [-1,1].
///
///The Tanh function is 
///\f[ f(x)=\tanh(x) = \frac 2 {1+exp^(-2x)}-1 \f]
///it's derivative can be computed as
///\f[ f'(x)= 1-f(x)^2 \f]
struct TanhNeuron{
	template<class Input, class Output>
	void function(Input const& input, Output& output)const{
		noalias(output) = tanh(input);
	}
	template<class Output, class Derivative>
	void multiplyDerivative(Output const& output, Derivative& der)const{
		noalias(der) *= typename Output::value_type(1) - sqr(output);
	}
};

///\brief Neuron which computes the Logistic (logistic) function with range [0,1].
///
///The Logistic function is 
///\f[ f(x)=\frac 1 {1+exp^(-x)}\f]
///it's derivative can be computed as
///\f[ f'(x)= f(x)(1-f(x)) \f]
struct LogisticNeuron{
	template<class Input, class Output>
	void function(Input const& input, Output& output)const{
		noalias(output) = sigmoid(input);
	}
	template<class Output, class Derivative>
	void multiplyDerivative(Output const& output, Derivative& der)const{
		noalias(der) *= output * (typename Output::value_type(1) - output);
	}
};

///\brief Fast sigmoidal function, which does not need to compute an exponential function.
///
///It is defined as
///\f[ f(x)=\frac x {1+|x|}\f]
///it's derivative can be computed as
///\f[ f'(x)= (1 - |f(x)|)^2 \f]
struct FastSigmoidNeuron{
	template<class Input, class Output>
	void function(Input const& input, Output& output)const{
		noalias(input)  = output/(typename Input::value_type(1)+abs(output));
	}
	template<class Output, class Derivative>
	void multiplyDerivative(Output const& output, Derivative& der)const{
		noalias(der) *= sqr(typename Output::value_type(1) - abs(output));
	}
};

///\brief Linear activation Neuron. 
struct LinearNeuron{
	template<class Input, class Output>
	void function(Input const& input, Output& output)const{
		noalias(output) = input;
	}
	template<class Output, class Derivative>
	void multiplyDerivative(Output const& output, Derivative& der)const{
		noalias(der) *= blas::repeat(typename Output::value_type(1), der.size1(), der.size2());
	}
};

///\brief Rectifier Neuron f(x) = max(0,x)
struct RectifierNeuron{
	template<class Input, class Output>
	void function(Input const& input, Output& output)const{
		noalias(output) = max(input,typename Input::value_type(0));
	}
	template<class Output, class Derivative>
	void multiplyDerivative(Output const& output, Derivative& der)const{
		//~ noalias(der) *= heaviside(output);
		for(std::size_t i = 0; i != output.size1(); ++i){
			for(std::size_t j = 0; j != output.size2(); ++j){
				der(i,j) *= output(i,j) > 0? 1.0:0.0;
			}
		}
	}
};


template <class NeuronType, class VectorType = RealVector>
class NeuronLayer : public AbstractModel<VectorType, VectorType, VectorType>{
private:
	typedef AbstractModel<VectorType,VectorType, VectorType> base_type;

	NeuronType m_neuron;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::ParameterVectorType ParameterVectorType;

	NeuronLayer(){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NeuronLayer"; }
	
	NeuronType const& neuron()const{ return m_neuron;}
	NeuronType& neuron(){ return m_neuron;}

	/// obtain the parameter vector
	ParameterVectorType parameterVector() const{
		return ParameterVectorType();
	}

	/// overwrite the parameter vector
	void setParameterVector(ParameterVectorType const& newParameters){
		SIZE_CHECK(newParameters.size() == 0);
	}

	/// return the number of parameter
	size_t numberOfParameters() const{
		return 0;
	}

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	using base_type::eval;

	void eval(BatchInputType const& inputs, BatchOutputType& outputs)const{
		outputs.resize(inputs.size1(),inputs.size2());
		m_neuron.function(inputs,outputs);
	}

	void eval(VectorType const& input, VectorType& output)const {
		output.resize(input.size());
		m_neuron.function(input,output);
	}
	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State& state)const{
		eval(inputs,outputs);
	}

	///\brief Calculates the first derivative w.r.t the parameters and summing them up over all patterns of the last computed batch
	void weightedParameterDerivative(
		BatchInputType const& patterns, 
		BatchOutputType const& outputs,
		BatchOutputType const& coefficients,
		State const& state, 
		ParameterVectorType& gradient
	)const{
		SIZE_CHECK(coefficients.size1()==patterns.size1());
		SIZE_CHECK(coefficients.size2()==patterns.size2());
	}
	///\brief Calculates the first derivative w.r.t the inputs and summs them up over all patterns of the last computed batch
	void weightedInputDerivative(
		BatchInputType const & patterns,
		BatchOutputType const & outputs,
		BatchOutputType const & coefficients,
		State const& state,
		BatchInputType& derivative
	)const{
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		SIZE_CHECK(coefficients.size2() == patterns.size2());

		derivative.resize(patterns.size1(),patterns.size2());
		noalias(derivative) = coefficients;
		m_neuron.multiplyDerivative(outputs, derivative);
		
	}

	/// From ISerializable
	void read(InArchive& archive){}
	/// From ISerializable
	void write(OutArchive& archive) const{}
};


}

#endif
