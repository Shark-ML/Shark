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
#ifndef SHARK_MODELS_LINEARMODEL_H
#define SHARK_MODELS_LINEARMODEL_H

#include <shark/Models/AbstractModel.h>
namespace shark {


///
/// \brief Linear Prediction
///
/// \par
/// A linear model makes predictions according to
/// \f$ y = f(x) = A x + b \f$. There are two important special cases:
/// The output may be a single number, and the offset term b may be
/// dropped.
///
/// The class allows for dense and sparse input vector types. However it assumes that 
/// the weight matrix and the ouputs are dense. There are some cases where this is not
/// good behavior. Check for example Normalizer for a class which is designed for sparse
/// inputs and outputs.
template <class InputType = RealVector>
class LinearModel : public AbstractModel<InputType,RealVector>
{
private:
	typedef AbstractModel<InputType,RealVector> base_type;
	typedef LinearModel<InputType> self_type;
	/// Wrapper for the type erasure
	RealMatrix m_matrix;
	RealVector m_offset;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;

	/// CDefault Constructor; use setStructure later
	LinearModel(){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}
	/// Constructor creating a model with given dimnsionalities and optional offset term.
	LinearModel(unsigned int inputs, unsigned int outputs = 1, bool offset = false)
	: m_matrix(outputs,inputs,0.0),m_offset(offset?outputs:0,0.0){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}
	///copy constructor
	LinearModel(LinearModel const& model)
	:m_matrix(model.m_matrix),m_offset(model.m_offset){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearModel"; }

	///swap
	friend void swap(LinearModel& model1,LinearModel& model2){
		swap(model1.m_matrix,model2.m_matrix);
		swap(model1.m_offset,model2.m_offset);
	}

	///operator =
	LinearModel& operator=(LinearModel const& model){
		self_type tempModel(model);
		swap(*this,tempModel);
		return *this;
	}

	/// Construction from matrix (and vector)
	LinearModel(RealMatrix const& matrix, RealVector const& offset = RealVector())
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
	RealVector parameterVector() const{
		RealVector ret(numberOfParameters());
		init(ret) << toVector(m_matrix),m_offset;
		
		return ret;
	}

	/// overwrite the parameter vector
	void setParameterVector(RealVector const& newParameters)
	{
		init(newParameters) >> toVector(m_matrix),m_offset;
	}

	/// return the number of parameter
	size_t numberOfParameters() const{
		return m_matrix.size1()*m_matrix.size2()+m_offset.size();
	}

	/// overwrite structure and parameters
	void setStructure(unsigned int inputs, unsigned int outputs = 1, bool offset = false){
		LinearModel<InputType> model(inputs,outputs,offset);
		swap(*this,model);
	}

	/// overwrite structure and parameters
	void setStructure(RealMatrix const& matrix, RealVector const& offset = RealVector()){
		m_matrix = matrix;
		m_offset = offset;
	}
	
	/// return a copy of the matrix in dense format
	RealMatrix const& matrix() const{
		return m_matrix;
	}
	
	RealMatrix& matrix(){
		return m_matrix;
	}

	/// return the offset
	RealVector const& offset() const{
		return m_offset;
	}
	RealVector& offset(){
		return m_offset;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	using base_type::eval;

	/// Evaluate the model: output = matrix * input + offset
	void eval(BatchInputType const& inputs, BatchOutputType& outputs)const{
		outputs.resize(inputs.size1(),m_matrix.size1());
		//we multiply with a set of row vectors from the left
		axpy_prod(inputs,trans(m_matrix),outputs);
		if (hasOffset()){
			noalias(outputs)+=repeat(m_offset,inputs.size1());
		}
	}
	/// Evaluate the model: output = matrix * input + offset
	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State& state)const{
		eval(inputs,outputs);
	}
	
	///\brief Calculates the first derivative w.r.t the parameters and summing them up over all patterns of the last computed batch 
	void weightedParameterDerivative(
		BatchInputType const& patterns, RealMatrix const& coefficients, State const& state, RealVector& gradient
	)const{
		SIZE_CHECK(coefficients.size2()==outputSize());
		SIZE_CHECK(coefficients.size1()==patterns.size1());

		gradient.resize(numberOfParameters());
		std::size_t inputs = inputSize();
		std::size_t outputs = outputSize();
		gradient.clear();

		blas::dense_matrix_adaptor<double> weightGradient = blas::adapt_matrix(outputs,inputs,gradient.storage());
		//sum_i coefficients(output,i)*pattern(i))
		axpy_prod(trans(coefficients),patterns,weightGradient,false);

		if (hasOffset()){
			std::size_t start = inputs*outputs;
			noalias(subrange(gradient, start, start + outputs)) = sum_rows(coefficients);
		}
	}
	///\brief Calculates the first derivative w.r.t the inputs and summs them up over all patterns of the last computed batch 
	void weightedInputDerivative(
		BatchInputType const & patterns,
		BatchOutputType const & coefficients,
		State const& state,
		BatchInputType& derivative
	)const{
		SIZE_CHECK(coefficients.size2() == outputSize());
		SIZE_CHECK(coefficients.size1() == patterns.size1());

		derivative.resize(patterns.size1(),inputSize());
		axpy_prod(coefficients,m_matrix,derivative);
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
