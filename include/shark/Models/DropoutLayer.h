/*!
 * 
 *
 * \brief       -
 *
 * \author      O.Krause
 * \date        2017
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
#ifndef MODELS_DROPOUTLAYER_H
#define MODELS_DROPOUTLAYER_H
 
#include <shark/Core/Random.h>
#include <shark/LinAlg/Base.h>
#include <shark/Models/AbstractModel.h>
namespace shark{

/// \brief Implements Dropout layer semantics
///
/// A dropout layer drops its input, i.e. sets it to 0 with a given probability. This is done for each element
/// separately. This means, model prodections are not deterministic any more. Thus, after training the 
/// output of several evaluations should be averaged. 
///
/// Dropout during training often leads to better regularized solutions in deep neural networks.
///
/// \ingroup models
template <class VectorType = RealVector>
class DropoutLayer : public AbstractModel<VectorType, VectorType, VectorType>{
private:
	typedef AbstractModel<VectorType,VectorType, VectorType> base_type;
	typedef blas::matrix<int, blas::row_major, typename VectorType::device_type> MatrixType;
	struct InternalState: public State{
		MatrixType mask;
	};
	Shape m_shape;
	random::rng_type* mep_rng;
	double m_dropoutProbability;
	
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::ParameterVectorType ParameterVectorType;

	DropoutLayer(Shape const& inputShape, double probability = 0.5, random::rng_type& rng = random::globalRng)
	: m_shape(inputShape), mep_rng(&rng), m_dropoutProbability(probability){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "DropoutLayer"; }

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
	
	///\brief Returns the expected shape of the input
	Shape inputShape() const{
		return m_shape;
	}
	///\brief Returns the shape of the output
	Shape outputShape() const{
		return m_shape;
	}

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}

	using base_type::eval;

	void eval(BatchInputType const& inputs, BatchOutputType& outputs)const{
		outputs.resize(inputs.size1(),inputs.size2());
		noalias(outputs) = inputs;
		for(std::size_t i = 0; i != outputs.size1(); ++i){
			for(std::size_t j = 0; j != outputs.size2(); ++j){
				if(!random::coinToss(*mep_rng,m_dropoutProbability)){
					outputs(i,j) = 0;
				}
			}
		}
	}

	void eval(VectorType const& input, VectorType& output)const {
		output.resize(input.size());
		noalias(output) = input;
		for(std::size_t j = 0; j != output.size(); ++j){
			if(!random::coinToss(*mep_rng,m_dropoutProbability)){
				output(j) = 0;
			}
		}
	}
	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State& state)const{
		MatrixType& mask = state.toState<InternalState>().mask;
		outputs.resize(inputs.size1(),inputs.size2());
		mask.resize(inputs.size1(),inputs.size2());
		for(std::size_t i = 0; i != outputs.size1(); ++i){
			for(std::size_t j = 0; j != outputs.size2(); ++j){
				mask(i,j) = random::coinToss(*mep_rng,m_dropoutProbability);
			}
		}
		noalias(outputs) = inputs * mask;
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

		MatrixType const& mask = state.toState<InternalState>().mask;
		derivative.resize(coefficients.size1(),coefficients.size2());
		noalias(derivative) = coefficients * mask;
	}

	/// From ISerializable
	void read(InArchive& archive){archive >> m_dropoutProbability;}
	/// From ISerializable
	void write(OutArchive& archive) const{ archive << m_dropoutProbability;}
};


}

#endif
