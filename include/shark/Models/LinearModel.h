/*!
*  \brief Implements a Model using a linear function.
*
*  \author  T. Glasmachers, O. Krause
*  \date    2010-2011
*
*  \par Copyright (c) 1999-2011:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-27974<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
*
*/
#ifndef SHARK_MODELS_LINEARMODEL_H
#define SHARK_MODELS_LINEARMODEL_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/Impl/LinearModel.inl>
#include <boost/scoped_ptr.hpp>
#include <boost/serialization/scoped_ptr.hpp>
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
/// \par
/// Under the hood the class allows for dense and sparse representations
/// of the matrix A. This is achieved by means of a type erasure.
///
template <class InputType = RealVector, class OutputType = InputType>
class LinearModel : public AbstractModel<InputType, OutputType>
{
private:
	typedef AbstractModel<InputType, OutputType> base_type;
	typedef LinearModel<InputType, OutputType> self_type;
	/// Wrapper for the type erasure
	boost::scoped_ptr<detail::LinearModelWrapperBase<InputType, OutputType> > mp_wrapper;

public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;

	/// Constructor of an invalid model; use setStructure later
	LinearModel(){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_SECOND_PARAMETER_DERIVATIVE;
	}
	///copy constructor
	LinearModel(const self_type& model):mp_wrapper(model.mp_wrapper->clone()){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_SECOND_PARAMETER_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearModel"; }

	///swap
	friend void swap(const LinearModel& model1,const LinearModel& model2){
		model1.mp_wrapper.swap(model2.mp_wrapper);
	}

	///operator =
	const self_type operator=(const self_type& model){
		self_type tempModel(model);
		swap(*this,tempModel);
	}

	/// Constructor
	LinearModel(unsigned int inputs, unsigned int outputs = 1, bool offset = false, bool sparse = false){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_SECOND_PARAMETER_DERIVATIVE;

		if (sparse) 
			mp_wrapper.reset(new detail::LinearModelWrapper<CompressedRealMatrix, InputType, OutputType>(inputs, outputs, offset));
		else 
			mp_wrapper.reset( new detail::LinearModelWrapper<RealMatrix, InputType, OutputType>(inputs, outputs, offset));
	}

	/// Construction from matrix
	LinearModel(RealMatrix const& matrix){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_SECOND_PARAMETER_DERIVATIVE;
		mp_wrapper.reset(new detail::LinearModelWrapper<RealMatrix, InputType, OutputType>(matrix));
	}

	/// Construction from matrix and vector
	LinearModel(RealMatrix const& matrix, OutputType offset){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_SECOND_PARAMETER_DERIVATIVE;
		mp_wrapper.reset(detail::LinearModelWrapper<RealMatrix, InputType, OutputType>(matrix, offset));
	}

	/// Construction from matrix
	LinearModel(CompressedRealMatrix const& matrix){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_SECOND_PARAMETER_DERIVATIVE;
		mp_wrapper.reset(new detail::LinearModelWrapper<CompressedRealMatrix, InputType, OutputType>(matrix));
	}

	/// Construction from matrix and vector
	LinearModel(CompressedRealMatrix const& matrix, RealVector offset){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_SECOND_PARAMETER_DERIVATIVE;
		mp_wrapper.reset(new detail::LinearModelWrapper<CompressedRealMatrix, InputType, OutputType>(matrix, offset));
	}

	/// check for the presence of an offset term
	bool hasOffset() const{
		SHARK_CHECK(mp_wrapper != NULL, "[LinearModel::hasOffset] model is not initialized");
		return mp_wrapper->hasOffset();
	}

	/// obtain the input dimension
	size_t inputSize() const{
		SHARK_CHECK(mp_wrapper != NULL, "[LinearModel::inputSize] model is not initialized");
		return mp_wrapper->inputSize();
	}

	/// obtain the output dimension
	size_t outputSize() const{
		SHARK_CHECK(mp_wrapper != NULL, "[LinearModel::outputSize] model is not initialized");
		return mp_wrapper->outputSize();
	}

	/// obtain the parameter vector
	RealVector parameterVector() const{
		SHARK_CHECK(mp_wrapper != NULL, "[LinearModel::parameterVector] model is not initialized");
		return mp_wrapper->parameterVector();
	}

	/// overwrite the parameter vector
	void setParameterVector(RealVector const& newParameters)
	{
		SHARK_CHECK(mp_wrapper != NULL, "[LinearModel::setParameterVector] model is not initialized");
		mp_wrapper->setParameterVector(newParameters);
	}

	/// return the number of parameter
	size_t numberOfParameters() const{
		SHARK_CHECK(mp_wrapper != NULL, "[LinearModel::numberOfParameters] model is not initialized");
		return mp_wrapper->numberOfParameters();
	}

	/// overwrite structure and parameters
	void setStructure(RealMatrix const& matrix){
		mp_wrapper.reset(new detail::LinearModelWrapper<RealMatrix, InputType, OutputType>(matrix));
	}

	/// overwrite structure and parameters
	void setStructure(unsigned int inputs, unsigned int outputs = 1, bool offset = false, bool sparse = false){
		if (sparse)
			mp_wrapper.reset(new detail::LinearModelWrapper<CompressedRealMatrix, InputType, OutputType>(inputs, outputs, offset));
		else
			mp_wrapper.reset( new detail::LinearModelWrapper<RealMatrix, InputType, OutputType>(inputs, outputs, offset));
	}

	/// overwrite structure and parameters
	void setStructure(RealMatrix const& matrix, const RealVector& offset){
		mp_wrapper.reset(new detail::LinearModelWrapper<RealMatrix, InputType, OutputType>(matrix, offset));
	}

	/// overwrite structure and parameters
	void setStructure(CompressedRealMatrix const& matrix){
		mp_wrapper.reset( new detail::LinearModelWrapper<CompressedRealMatrix, InputType, OutputType>(matrix));
	}

	/// overwrite structure and parameters
	void setStructure(CompressedRealMatrix const& matrix,const RealVector& offset){
		SHARK_CHECK(matrix.size1() == offset.size(), "[LinearModel::setStructure] dimension mismatch between matrix and offset");
		mp_wrapper.reset(new detail::LinearModelWrapper<CompressedRealMatrix, InputType, OutputType>(matrix, offset));
	}

	/// return a copy of the matrix in dense format
	RealMatrix matrix() const{
		SHARK_CHECK(mp_wrapper != NULL, "[LinearModel::matrix] model is not initialized");
		RealMatrix ret; mp_wrapper->matrix(ret); return ret;
	}

	/// return the offset
	OutputType const& offset() const{
		SHARK_CHECK(mp_wrapper != NULL, "[LinearModel::offset] model is not initialized");
		return mp_wrapper->offset();
	}

	/// return a copy of a row of the matrix in dense format
	RealVector matrixRow(size_t index) const{
		SHARK_CHECK(mp_wrapper != NULL, "[LinearModel::matrixRow] model is not initialized");
		RealVector ret;
		mp_wrapper->matrixRow(index, ret);
		return ret;
	}

	/// return a copy of a column of the matrix in dense format
	RealVector matrixColumn(size_t index) const{
		SHARK_CHECK(mp_wrapper != NULL, "[LinearModel::matrixColumn] model is not initialized");
		RealVector ret;
		mp_wrapper->matrixColumn(index, ret);
		return ret;
	}
	
	boost::shared_ptr<State> createState()const{
		SHARK_CHECK(mp_wrapper != NULL, "[LinearModel::createState] model is not initialized");
		return mp_wrapper->createState();
	}

	using base_type::eval;

	/// Evaluate the model: output = matrix * input + offset
	void eval(BatchInputType const& input, BatchOutputType& output)const{
		SHARK_CHECK(mp_wrapper != NULL, "[LinearModel::eval] model is not initialized");
		mp_wrapper->eval(input, output);
	}
	/// Evaluate the model: output = matrix * input + offset
	void eval(BatchInputType const& input, BatchOutputType& output, State& state)const{
		SHARK_CHECK(mp_wrapper != NULL, "[LinearModel::eval] model is not initialized");
		mp_wrapper->eval(input, output,state);
	}
	
	///\brief calculates the first derivative w.r.t the parameters and summing them up over all patterns of the last computed batch 
	void weightedParameterDerivative(
		BatchInputType const& pattern, RealMatrix const& coefficients, State const& state, RealVector& gradient
	)const{
		mp_wrapper->weightedParameterDerivative(pattern, coefficients, state, gradient);
	}
	void weightedParameterDerivative(
		BatchInputType const & patterns,
		BatchOutputType const & coefficients,
		Batch<RealMatrix>::type const & errorHessian,//maybe a batch of matrices is bad?,
		State const& state,
		RealVector& derivative,
		RealMatrix& hessian
	)const{
		mp_wrapper->weightedParameterDerivative(patterns, coefficients, errorHessian, state, derivative, hessian);
	}

	/// From ISerializable
	void read(InArchive& archive){
		//let's hope, noone will ever create a templated setStructure method, in this case, serialization _must_ fail.
		archive.register_type<detail::LinearModelWrapper<CompressedRealMatrix, InputType, OutputType> >();
		archive.register_type<detail::LinearModelWrapper<RealMatrix, InputType, OutputType> >();
		archive & mp_wrapper;
	}

	/// From ISerializable
	void write(OutArchive& archive) const{
		archive.register_type<detail::LinearModelWrapper<CompressedRealMatrix, InputType, OutputType> >();
		archive.register_type<detail::LinearModelWrapper<RealMatrix, InputType, OutputType> >();
		archive & mp_wrapper;
	}
};


}
#endif
