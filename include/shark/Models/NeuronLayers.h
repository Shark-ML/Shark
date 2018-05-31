/*!
 * 
 *
 * \file
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
	

/// \defgroup activations Neuron activation functions
/// \ingroup models
/// Neuron activation functions used for neural network nonlinearities.

	
///\brief Neuron which computes the hyperbolic tangenst with range [-1,1].
///
///The Tanh function is 
///\f[ f(x)=\tanh(x) = \frac 2 {1+exp^(-2x)}-1 \f]
///it's derivative can be computed as
///\f[ f'(x)= 1-f(x)^2 \f]
///
/// \ingroup activations
struct TanhNeuron{
	typedef EmptyState State;
	template<class Arg>
	void evalInPlace(Arg& arg)const{
		noalias(arg) = tanh(arg);
	}
	
	template<class Arg>
	void evalInPlace(Arg& arg, State&)const{
		evalInPlace(arg);
	}
	
	template<class Output, class Derivative>
	void multiplyDerivative(Output const& output, Derivative& der, State const& )const{
		noalias(der) *= typename Output::value_type(1) - sqr(output);
	}
};

///\brief Neuron which computes the Logistic (logistic) function with range [0,1].
///
///The Logistic function is 
///\f[ f(x)=\frac 1 {1+exp^(-x)}\f]
///it's derivative can be computed as
///\f[ f'(x)= f(x)(1-f(x)) \f]
///
/// \ingroup activations
struct LogisticNeuron{
	typedef EmptyState State;
	template<class Arg>
	void evalInPlace(Arg& arg)const{
		noalias(arg) = sigmoid(arg);
	}
	
	template<class Arg>
	void evalInPlace(Arg& arg, State&)const{
		evalInPlace(arg);
	}

	template<class Output, class Derivative>
	void multiplyDerivative(Output const& output, Derivative& der, State const& state)const{
		noalias(der) *= output * (typename Output::value_type(1) - output);
	}
};

///\brief Fast sigmoidal function, which does not need to compute an exponential function.
///
///It is defined as
///\f[ f(x)=\frac x {1+|x|}\f]
///it's derivative can be computed as
///\f[ f'(x)= (1 - |f(x)|)^2 \f]
///
/// \ingroup activations
struct FastSigmoidNeuron{
	typedef EmptyState State;
	template<class Arg>
	void evalInPlace(Arg& arg)const{
		noalias(arg)  /= typename Arg::value_type(1)+abs(arg);
	}
	
	template<class Arg>
	void evalInPlace(Arg& arg, State&)const{
		evalInPlace(arg);
	}

	template<class Output, class Derivative>
	void multiplyDerivative(Output const& output, Derivative& der, State const& state)const{
		noalias(der) *= sqr(typename Output::value_type(1) - abs(output));
	}
};

///\brief Linear activation Neuron. 
///
///It is defined as
///\f[ f(x)=x\f]
///
/// \ingroup activations
struct LinearNeuron{
	typedef EmptyState State;
	template<class Arg>
	void evalInPlace(Arg&)const{}
	
	template<class Arg>
	void evalInPlace(Arg& arg, State const&)const{}
	
	template<class Output, class Derivative>
	void multiplyDerivative(Output const& output, Derivative& der, State const& state)const{}
};

///\brief Rectifier Neuron f(x) = max(0,x)
///
/// \ingroup activations
struct RectifierNeuron{
	typedef EmptyState State;
	template<class Arg>
	void evalInPlace(Arg& arg)const{
		noalias(arg) = max(arg,typename Arg::value_type(0));
	}
	
	template<class Arg>
	void evalInPlace(Arg& arg, State&)const{
		evalInPlace(arg);
	}
	
	template<class Output, class Derivative>
	void multiplyDerivative(Output const& output, Derivative& der, State const& state)const{
		//~ noalias(der) *= heaviside(output);
		//~ for(std::size_t i = 0; i != output.size1(); ++i){
			//~ for(std::size_t j = 0; j != output.size2(); ++j){
				//~ der(i,j) *= output(i,j) > 0? 1.0:0.0;
			//~ }
		//~ }
		noalias(der) *= output > 0;
	}
};

///\brief Normalizes the sum of inputs to one.
///
/// \f[ f_i(x)= x_i \ \sum_j x_j \f]
/// Normalization will reinterpret the input as probabilities. Therefore no negative valeus are allowed.
///
/// \ingroup activations
template<class VectorType = RealVector>
struct NormalizerNeuron{
	struct State: public shark::State{
		VectorType norm;
		
		void resize(std::size_t patterns){
			norm.resize(patterns);
		}
	};
	
	template<class Arg, class Device>
	void evalInPlace(blas::vector_expression<Arg,Device>& arg)const{
		noalias(arg) /= sum(arg);
	}
	
	template<class Arg, class Device>
	void evalInPlace(blas::matrix_expression<Arg,Device>& arg)const{
		noalias(trans(arg)) /= blas::repeat(sum(as_rows(arg)),arg().size2());
	}
	
	template<class Arg, class Device>
	void evalInPlace(blas::matrix_expression<Arg,Device>& arg, State& state)const{
		state.norm.resize(arg().size1());
		noalias(state.norm) = sum(as_rows(arg));
		noalias(arg) /= trans(blas::repeat(state.norm,arg().size2()));
	}
	
	template<class Output, class Derivative>
	void multiplyDerivative(Output const& output, Derivative& der, State const& s)const{
		for(std::size_t i = 0; i != output.size1(); ++i){
			double constant=inner_prod(row(der,i),row(output,i));
			noalias(row(der,i))= (row(der,i)-constant)/s.norm(i);
		}
	}
};

///\brief Computes the softmax activation function.
///
/// \f[ f_i(x)= \exp(x_i) \ \sum_j \exp(x_j) \f]
///
/// computes the exponential function of the inputs and normalizes the outputs to sum to one. This is 
/// the NormalizerNeuron just without the constraint of values being positive
///
/// \ingroup activations
template<class VectorType = RealVector>
struct SoftmaxNeuron{
	typedef EmptyState State;
	
	template<class Arg, class Device>
	void evalInPlace(blas::vector_expression<Arg,Device>& arg)const{
		noalias(arg) = exp(arg);
		noalias(arg) /= sum(arg);
	}
	
	template<class Arg, class Device>
	void evalInPlace(blas::matrix_expression<Arg,Device>& arg)const{
		noalias(arg) = exp(arg);
		noalias(arg) /= trans(blas::repeat(sum(as_rows(arg)),arg().size2()));
	}
	
	template<class Arg, class Device>
	void evalInPlace(blas::matrix_expression<Arg,Device>& arg, State&)const{
		evalInPlace(arg);
	}
	
	template<class Output, class Derivative>
	void multiplyDerivative(Output const& output, Derivative& der, State const& s)const{
		auto mass = eval_block(sum(as_rows(der * output)));
		noalias(der) -= trans(blas::repeat(mass, der.size2()));
		noalias(der) *= output;
	}
};



///\brief Neuron activation layer.
///
/// Applies a nonlinear activation function to the given input. Various choices for activations
/// are given in \ref activations.
///
/// \ingroup models
template <class NeuronType, class VectorType = RealVector>
class NeuronLayer : public AbstractModel<VectorType, VectorType, VectorType>{
private:
	typedef AbstractModel<VectorType,VectorType, VectorType> base_type;

	NeuronType m_neuron;
	Shape m_shape;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::ParameterVectorType ParameterVectorType;

	NeuronLayer(Shape const& shape = Shape()): m_shape(shape){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NeuronLayer"; }
	
	NeuronType const& neuron()const{ return m_neuron;}
	NeuronType& neuron(){ return m_neuron;}
	
	Shape inputShape() const{
		return m_shape;
	}
	
	Shape outputShape() const{
		return m_shape;
	}

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
		return boost::shared_ptr<State>(new typename NeuronType::State());
	}

	using base_type::eval;

	void eval(BatchInputType const& inputs, BatchOutputType& outputs)const{
		SIZE_CHECK(inputs.size2() == m_shape.numElements());
		outputs.resize(inputs.size1(),inputs.size2());
		noalias(outputs) = inputs;
		m_neuron.evalInPlace(outputs);
	}

	void eval(VectorType const& input, VectorType& output)const{
		SIZE_CHECK(input.size() == m_shape.numElements());
		output.resize(input.size());
		noalias(output) = input;
		m_neuron.evalInPlace(output);
	}
	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State& state)const{
		SIZE_CHECK(inputs.size2() == m_shape.numElements());
		outputs.resize(inputs.size1(),inputs.size2());
		noalias(outputs) = inputs;
		m_neuron.evalInPlace(outputs, state.toState<typename NeuronType::State>());
	}

	///\brief Calculates the first derivative w.r.t the parameters and summing them up over all inputs of the last computed batch
	void weightedParameterDerivative(
		BatchInputType const& inputs, 
		BatchOutputType const& outputs,
		BatchOutputType const& coefficients,
		State const& state, 
		ParameterVectorType& gradient
	)const{
		SIZE_CHECK(coefficients.size1()==inputs.size1());
		SIZE_CHECK(coefficients.size2()==inputs.size2());
	}
	///\brief Calculates the first derivative w.r.t the inputs and summs them up over all inputs of the last computed batch
	void weightedInputDerivative(
		BatchInputType const & inputs,
		BatchOutputType const & outputs,
		BatchOutputType const & coefficients,
		State const& state,
		BatchInputType& derivative
	)const{
		SIZE_CHECK(coefficients.size1() == inputs.size1());
		SIZE_CHECK(coefficients.size2() == inputs.size2());

		derivative.resize(inputs.size1(),inputs.size2());
		noalias(derivative) = coefficients;
		m_neuron.multiplyDerivative(outputs, derivative, state.toState<typename NeuronType::State>());
		
	}

	/// From ISerializable
	void read(InArchive& archive){ archive >> m_shape;}
	/// From ISerializable
	void write(OutArchive& archive) const{ archive << m_shape;}
};


}

#endif
