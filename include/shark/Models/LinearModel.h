/*!
 *
 *
 * \brief       Implements a Model using a linear function.
 *
 *
 *
 * \author      T. Glasmachers, O. Krause
 * \date        2010-2011
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
#ifndef SHARK_MODELS_LINEARMODEL_H
#define SHARK_MODELS_LINEARMODEL_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/NeuronLayers.h>
namespace shark {


///
/// \brief Linear Prediction with optional activation function
///
/// \par
/// This model computes the result of
/// \f$ y = f(x) = g(A x + b) \f$, where g is an arbitrary activation function.
/// By default g is the identity and the model is a simple linear model. 
/// Otherwise, this is known as a generalized linear model. There are two important special cases:
/// The output may be a single number, and the offset term b may be
/// dropped.
///
/// The class allows for dense and sparse input vector types. However it assumes that
/// the weight matrix and the ouputs are dense. There are some cases where this is not
/// good behavior. Check for example Normalizer for a class which is designed for sparse
/// inputs and outputs.
template <class InputType = RealVector, class ActivationFunction = LinearNeuron>
class LinearModel : public AbstractModel<
	InputType,
	blas::vector<typename InputType::value_type, typename InputType::device_type>,//type of output uses same device and precision as input
	blas::vector<typename InputType::value_type, typename InputType::device_type>//type of parameters uses same device and precision as input
>{
private:
	typedef blas::vector<typename InputType::value_type, typename InputType::device_type> VectorType;
	typedef blas::matrix<typename InputType::value_type, blas::row_major, typename InputType::device_type> MatrixType;
	typedef AbstractModel<InputType,VectorType, VectorType> base_type;
	typedef LinearModel<InputType, ActivationFunction> self_type;
	MatrixType m_matrix;
	VectorType m_offset;
	ActivationFunction m_activation;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;//same as MatrixType
	typedef typename base_type::ParameterVectorType ParameterVectorType;//same as VectorType

	/// CDefault Constructor; use setStructure later
	LinearModel(){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		if(std::is_same<typename InputType::storage_type::storage_tag, blas::dense_tag>::value){
			base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		}
	}
	/// Constructor creating a model with given dimnsionalities and optional offset term.
	LinearModel(std::size_t inputs, std::size_t outputs = 1, bool offset = false)
	: m_matrix(outputs,inputs,0.0),m_offset(offset?outputs:0,0.0){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearModel"; }

	/// Construction from matrix (and vector)
	LinearModel(MatrixType const& matrix, VectorType const& offset = VectorType())
	:m_matrix(matrix),m_offset(offset){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}

	/// check for the presence of an offset term
	bool hasOffset() const{
		return m_offset.size() != 0;
	}

	/// obtain the input dimension
	size_t inputSize() const{
		return m_matrix.size2();
	}

	/// obtain the output dimension
	size_t outputSize() const{
		return m_matrix.size1();
	}

	/// obtain the parameter vector
	ParameterVectorType parameterVector() const{
		return to_vector(m_matrix) | m_offset;
	}

	/// overwrite the parameter vector
	void setParameterVector(ParameterVectorType const& newParameters){
		noalias(to_vector(m_matrix)) = subrange(newParameters,0,inputSize() * outputSize());
		noalias(m_offset) = subrange(newParameters,inputSize() * outputSize(),newParameters.size());
	}

	/// return the number of parameter
	size_t numberOfParameters() const{
		return m_matrix.size1()*m_matrix.size2()+m_offset.size();
	}

	/// overwrite structure and parameters
	void setStructure(std::size_t inputs, std::size_t outputs = 1, bool offset = false){
		LinearModel<InputType, ActivationFunction> model(inputs,outputs,offset);
		*this = model;
	}

	/// overwrite structure and parameters
	void setStructure(MatrixType const& matrix, VectorType const& offset = VectorType()){
		m_matrix = matrix;
		m_offset = offset;
	}

	/// return a copy of the matrix in dense format
	MatrixType const& matrix() const{
		return m_matrix;
	}

	MatrixType& matrix(){
		return m_matrix;
	}

	/// return the offset
	VectorType const& offset() const{
		return m_offset;
	}
	VectorType& offset(){
		return m_offset;
	}
	
	/// \brief Returns the activation function.
	ActivationFunction const& activationFunction()const{
		return m_activation;
	}
	
	/// \brief Returns the activation function.
	ActivationFunction& activationFunction(){
		return m_activation;
	}

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new typename ActivationFunction::State());
	}

	using base_type::eval;

	/// Evaluate the model: output = matrix * input + offset
	void eval(BatchInputType const& inputs, BatchOutputType& outputs)const{
		outputs.resize(inputs.size1(),m_matrix.size1());
		//we multiply with a set of row vectors from the left
		noalias(outputs) = inputs % trans(m_matrix);
		if (hasOffset()){
			noalias(outputs)+=repeat(m_offset,inputs.size1());
		}
		m_activation.evalInPlace(outputs);
	}

	void eval(InputType const& input, VectorType& output)const {
		output.resize(m_matrix.size1());
		//we multiply with a set of row vectors from the left
		noalias(output) = m_matrix % input;
		if (hasOffset()) {
			noalias(output) += m_offset;
		}
		m_activation.evalInPlace(output);
	}
	/// Evaluate the model: output = matrix * input + offset
	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State& state)const{
		outputs.resize(inputs.size1(),m_matrix.size1());
		//we multiply with a set of row vectors from the left
		noalias(outputs) = inputs % trans(m_matrix);
		if (hasOffset()){
			noalias(outputs)+=repeat(m_offset,inputs.size1());
		}
		m_activation.evalInPlace(outputs, state.toState<typename ActivationFunction::State>());
	}

	///\brief Calculates the first derivative w.r.t the parameters and summing them up over all patterns of the last computed batch
	void weightedParameterDerivative(
		BatchInputType const& patterns,
		BatchOutputType const& outputs,
		BatchOutputType const& coefficients,
		State const& state,
		ParameterVectorType& gradient
	)const{
		SIZE_CHECK(coefficients.size2()==outputSize());
		SIZE_CHECK(coefficients.size1()==patterns.size1());

		gradient.resize(numberOfParameters());
		std::size_t numInputs = inputSize();
		std::size_t numOutputs = outputSize();
		gradient.clear();
		std::size_t matrixParams = numInputs*numOutputs;

		auto weightGradient = blas::to_matrix(subrange(gradient,0,matrixParams), numOutputs,numInputs);
		
		BatchOutputType delta = coefficients;
		m_activation.multiplyDerivative(outputs,delta, state.toState<typename ActivationFunction::State>());
		//sum_i coefficients(output,i)*pattern(i))
		noalias(weightGradient) = trans(delta) % patterns;

		if (hasOffset()){
			noalias(subrange(gradient, matrixParams, matrixParams + numOutputs)) = sum_rows(delta);
		}
	}
	///\brief Calculates the first derivative w.r.t the inputs and summs them up over all patterns of the last computed batch
	void weightedInputDerivative(
		BatchInputType const & patterns,
		BatchOutputType const& outputs,
		BatchOutputType const & coefficients,
		State const& state,
		BatchInputType& derivative
	)const{
		SIZE_CHECK(coefficients.size2() == outputSize());
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		
		//compute chain rule
		BatchOutputType delta = coefficients;
		m_activation.multiplyDerivative(outputs,delta, state.toState<typename ActivationFunction::State>());
		
		derivative.resize(patterns.size1(),inputSize());
		noalias(derivative) = MatrixType(delta % m_matrix);//TODO: bug in remora will lead to compile error if derivative is sparse
	}
	
	void weightedDerivatives(
		BatchInputType const & patterns,
		BatchOutputType const& outputs,
		BatchOutputType const & coefficients,
		State const& state,
		RealVector& parameterDerivative,
		BatchInputType& inputDerivative
	)const{
		SIZE_CHECK(coefficients.size2()==outputSize());
		SIZE_CHECK(coefficients.size1()==patterns.size1());

		std::size_t numInputs = inputSize();
		std::size_t numOutputs = outputSize();
		
		//compute chain rule
		BatchOutputType delta = coefficients;
		m_activation.multiplyDerivative(outputs,delta, state.toState<typename ActivationFunction::State>());
		
		//compute input derivative
		inputDerivative.resize(patterns.size1(),numInputs);
		noalias(inputDerivative) = MatrixType(delta % m_matrix);
		
		//compute parameter derivative
		parameterDerivative.resize(numberOfParameters());
		parameterDerivative.clear();
		std::size_t matrixParams = numInputs*numOutputs;
		auto weightGradient = blas::to_matrix(subrange(parameterDerivative,0,matrixParams), numOutputs,numInputs);
		auto offsetGradient = subrange(parameterDerivative,matrixParams,parameterDerivative.size());
		
		//sum_i coefficients(output,i)*pattern(i))
		noalias(weightGradient) = trans(delta) % patterns;
		if (hasOffset()){
			noalias(offsetGradient) = sum_rows(delta);
		}
	}

	/// From ISerializable
	void read(InArchive& archive){
		archive >> m_matrix;
		archive >> m_offset;
	}
	/// From ISerializable
	void write(OutArchive& archive) const{
		archive << m_matrix;
		archive << m_offset;
	}
};


}
#endif
