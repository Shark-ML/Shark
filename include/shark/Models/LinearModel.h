/*!
 *
 *
 * \brief       Implements a Model using a linear function.
 *
 *
 *
 * \author      T. Glasmachers, O. Krause
 * \date        2010-2017
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
#include <shark/Models/Classifier.h>
namespace shark {


///
/// \brief Linear Prediction with optional activation function
///
/// \par
/// This model computes the result of
/// \f$ y = f(x) = g(A x + b) \f$, where g is an arbitrary activation function, see \ref activations..
/// By default g is the identity and the model is a simple linear model. 
/// Otherwise, this is known as a generalized linear model. There are two important special cases:
/// The output may be a single number, and the offset term b may be
/// dropped.
///
/// The class allows for dense and sparse input vector types. However it assumes that
/// the weight matrix and the ouputs are dense. There are some cases where this is not
/// good behavior. Check for example Normalizer for a class which is designed for sparse
/// inputs and outputs.
///
/// \ingroup models
template <class InputType = RealVector, class ActivationFunction = LinearNeuron>
class LinearModel : public AbstractModel<
	InputType,
	blas::vector<typename InputType::value_type, typename InputType::device_type>,//type of output uses same device and precision as input
	blas::vector<typename InputType::value_type, typename InputType::device_type>//type of parameters uses same device and precision as input
>{
public:
	typedef blas::vector<typename InputType::value_type, typename InputType::device_type> VectorType;
	typedef blas::matrix<typename InputType::value_type, blas::row_major, typename InputType::device_type> MatrixType;
private:	
	typedef AbstractModel<InputType,VectorType, VectorType> base_type;
	typedef LinearModel<InputType, ActivationFunction> self_type;
	Shape m_inputShape;
	Shape m_outputShape;
	MatrixType m_matrix;
	VectorType m_offset;
	ActivationFunction m_activation;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;//same as MatrixType
	typedef typename base_type::ParameterVectorType ParameterVectorType;//same as VectorType

	/// CDefault Constructor; use setStructure later
	LinearModel(){
		this->m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		if(std::is_base_of<blas::dense_tag, typename InputType::storage_type::storage_tag>::value){
			this->m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		}
	}
	/// Constructor creating a model with given dimensionalities and optional offset term.
	LinearModel(Shape const& inputs, Shape const& outputs = 1, bool offset = false)
	: m_inputShape(inputs)
	, m_outputShape(outputs)
	, m_matrix(outputs.numElements(),inputs.numElements(),0.0)
	, m_offset(offset?outputs.numElements():0,0.0){
		this->m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		if(std::is_base_of<blas::dense_tag, typename InputType::storage_type::storage_tag>::value){
			this->m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		}
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearModel"; }

	/// Construction from matrix (and vector)
	LinearModel(MatrixType const& matrix, VectorType const& offset = VectorType())
	: m_inputShape(matrix.size2())
	, m_outputShape(matrix.size1())
	, m_matrix(matrix)
	, m_offset(offset){
		this->m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		if(std::is_base_of<blas::dense_tag, typename InputType::storage_type::storage_tag>::value){
			this->m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		}
	}

	/// check for the presence of an offset term
	bool hasOffset() const{
		return m_offset.size() != 0;
	}

	///\brief Returns the expected shape of the input
	Shape inputShape() const{
		return m_inputShape;
	}
	///\brief Returns the shape of the output
	Shape outputShape() const{
		return m_outputShape;
	}

	/// obtain the parameter vector
	ParameterVectorType parameterVector() const{
		return to_vector(m_matrix) | m_offset;
	}

	/// overwrite the parameter vector
	void setParameterVector(ParameterVectorType const& newParameters){
		std::size_t numInputs = inputShape().numElements();
		std::size_t numOutputs = outputShape().numElements();
		noalias(to_vector(m_matrix)) = subrange(newParameters, 0, numInputs * numOutputs);
		noalias(m_offset) = subrange(newParameters, numInputs * numOutputs, newParameters.size());
	}

	/// return the number of parameter
	size_t numberOfParameters() const{
		return m_matrix.size1()*m_matrix.size2()+m_offset.size();
	}

	/// overwrite structure and parameters
	void setStructure(Shape const& inputs, Shape const& outputs = 1, bool offset = false){
		LinearModel<InputType, ActivationFunction> model(inputs,outputs,offset);
		*this = model;
	}

	/// overwrite structure and parameters
	void setStructure(MatrixType const& matrix, VectorType const& offset = VectorType()){
		LinearModel<InputType, ActivationFunction> model(matrix,offset);
		*this = model;
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
		SIZE_CHECK(coefficients.size2()==m_matrix.size1());
		SIZE_CHECK(coefficients.size1()==patterns.size1());

		gradient.resize(numberOfParameters());
		std::size_t numInputs = inputShape().numElements();
		std::size_t numOutputs = outputShape().numElements();
		gradient.clear();
		std::size_t matrixParams = numInputs*numOutputs;

		auto weightGradient = blas::to_matrix(subrange(gradient,0,matrixParams), numOutputs,numInputs);
		
		BatchOutputType delta = coefficients;
		m_activation.multiplyDerivative(outputs,delta, state.toState<typename ActivationFunction::State>());
		//sum_i coefficients(output,i)*pattern(i))
		noalias(weightGradient) = trans(delta) % patterns;

		if (hasOffset()){
			noalias(subrange(gradient, matrixParams, matrixParams + numOutputs)) = sum(as_columns(delta));
		}
	}
	///\brief Calculates the first derivative w.r.t the inputs and summs them up over all patterns of the last computed batch
	void weightedInputDerivative(
		BatchInputType const & patterns,
		BatchOutputType const& outputs,
		BatchOutputType const & coefficients,
		State const& state,
		MatrixType& derivative
	)const{
		SIZE_CHECK(coefficients.size2() == m_matrix.size1());
		SIZE_CHECK(coefficients.size1() == patterns.size1());
		
		//compute chain rule
		BatchOutputType delta = coefficients;
		m_activation.multiplyDerivative(outputs,delta, state.toState<typename ActivationFunction::State>());
		
		derivative.resize(patterns.size1(),patterns.size2());
		noalias(derivative) = delta % m_matrix;
	}
	
	void weightedDerivatives(
		BatchInputType const & patterns,
		BatchOutputType const& outputs,
		BatchOutputType const & coefficients,
		State const& state,
		ParameterVectorType& parameterDerivative,
		MatrixType& inputDerivative
	)const{
		SIZE_CHECK(coefficients.size2()==m_matrix.size1());
		SIZE_CHECK(coefficients.size1()==patterns.size1());

		std::size_t numInputs = inputShape().numElements();
		std::size_t numOutputs = outputShape().numElements();
		
		//compute chain rule
		BatchOutputType delta = coefficients;
		m_activation.multiplyDerivative(outputs,delta, state.toState<typename ActivationFunction::State>());
		
		//compute input derivative
		inputDerivative.resize(patterns.size1(),numInputs);
		noalias(inputDerivative) = delta % m_matrix;
		
		//compute parameter derivative
		parameterDerivative.resize(numberOfParameters());
		parameterDerivative.clear();
		std::size_t matrixParams = numInputs*numOutputs;
		auto weightGradient = blas::to_matrix(subrange(parameterDerivative,0,matrixParams), numOutputs,numInputs);
		auto offsetGradient = subrange(parameterDerivative,matrixParams,parameterDerivative.size());
		
		//sum_i coefficients(output,i)*pattern(i))
		noalias(weightGradient) = trans(delta) % patterns;
		if (hasOffset()){
			noalias(offsetGradient) = sum(as_columns(delta));
		}
	}

	/// From ISerializable
	void read(InArchive& archive){
		archive >> m_matrix;
		archive >> m_offset;
		archive >> m_inputShape;
		archive >> m_outputShape;
	}
	/// From ISerializable
	void write(OutArchive& archive) const{
		archive << m_matrix;
		archive << m_offset;
		archive << m_inputShape;
		archive << m_outputShape;
	}
};

/// \brief Basic linear classifier.
///
/// The LinearClassifier class is a multi class classifier model
/// suited for linear discriminant analysis. For c classes
/// \f$ 0, \dots, c-1 \f$  the model computes
///  
/// \f$ \arg \max_i w_i^T x + b_i \f$
/// 
/// Thus is it a linear model with arg max computation.
/// The internal linear model can be queried using decisionFunction().
///
/// \ingroup models
template<class VectorType = RealVector>
class LinearClassifier : public Classifier<LinearModel<VectorType> >
{
public:
	typedef typename LinearModel<VectorType>::MatrixType MatrixType;
	LinearClassifier(){}
	
	/// Constructor creating a model with given dimensionalities and optional offset term.
	LinearClassifier(Shape const& inputs, std::size_t numClasses, bool offset = false){
		setStructure(inputs, numClasses, offset);
	}
	
	/// Constructor from weight matrix (and optional offset).
	LinearClassifier(MatrixType const& matrix, VectorType const& offset = VectorType()){
		setStructure(matrix, offset);
	}

	std::string name() const
	{ return "LinearClassifier"; }
	
	/// overwrite structure and parameters
	void setStructure(Shape const& inputs, std::size_t numClasses, bool offset = false){
		this->decisionFunction().setStructure(inputs, numClasses, offset);
	}

	/// overwrite structure and parameters
	void setStructure(MatrixType const& matrix, VectorType const& offset = VectorType()){
		this->decisionFunction().setStructure(matrix, offset);
	}
};

}
#endif
