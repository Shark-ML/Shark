/*!
 *
 *
 * \brief       Implements a model applying a convolution to an image
 *
 *
 *
 * \author    
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
#ifndef SHARK_MODELS_CONV2DModel_H
#define SHARK_MODELS_CONV2DModel_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/NeuronLayers.h>
#include <shark/Core/Images/Enums.h>
#include <shark/Core/Images/Convolution.h>
namespace shark {

///
/// \brief Convolutional Model for 2D image data.
///
/// \par
/// This model computes the result of
/// \f$ y = f(x) = g(\text{convolution}(w, x) + b) \f$, where g is an arbitrary activation function  \ref activations and
/// convolution is the convolution of the input image x with the filters w.  b is a vector with one entry for each filter which is then applied to each response above
///
/// The image is allowed to have several channels and are linearized to a single vector of size width * height * numChannels.
/// This is done by itnerleaving channels, i.e. for a pixel all channels are stored contiguously. Then the pixels are stored in
/// a row-major scheme.
///
/// For handling edge condition, the Conv2D model handles two different convolution modes:
///
/// Padding::Valid:
/// The output is only computed on patches which are fully inside the unpadded image as a linearized vector in the same format
/// of size (width - filter_width+1) * (height - filter_height+1) * numFilters.
///
/// Padding::ZeroPad
/// The output input is padded with zeros and the output has the same size as the input
/// of size width * height * numFilters.
///
/// \ingroup models
template <class VectorType = RealVector, class ActivationFunction = LinearNeuron>
class Conv2DModel : public AbstractModel<VectorType,VectorType,VectorType>{
private:
	typedef AbstractModel<VectorType,VectorType, VectorType> base_type;
	typedef Conv2DModel<VectorType, ActivationFunction> self_type;
	typedef typename VectorType::value_type value_type;
	typedef typename VectorType::device_type device_type;
	typedef blas::matrix<value_type, blas::row_major,device_type> MatrixType;
	static_assert(!std::is_same<typename VectorType::storage_type::storage_tag, blas::dense_tag>::value, "Conv2D not implemented for sparse inputs");
public:
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ParameterVectorType ParameterVectorType;

	/// Default Constructor; use setStructure later.
	Conv2DModel(){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}
	
	///\brief Sets the structure by setting the dimensionalities of image and filters.
	///
	/// \arg imageShape Shape of the image imHeight x imWidth x channel
	/// \arg filterShape Shape of the filter matrix numFilters x fiHeight x fiWidth x channel
	/// \arg type Type of convolution padding to perform
	Conv2DModel(
		Shape const& imageShape, Shape const& filterShape, Padding type = Padding::ZeroPad
	){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		setStructure(imageShape, filterShape, type);
	}

	std::string name() const
	{ return "Conv2DModel"; }
	
	///\brief Returns the expected shape of the input
	Shape inputShape() const{
		return m_inputShape;
	}
	///\brief Returns the shape of the output
	Shape outputShape() const{
		return m_outputShape;
	}
	
	/// \brief Returns the activation function.
	ActivationFunction const& activationFunction()const{
		return m_activation;
	}
	
	/// \brief Returns the activation function.
	ActivationFunction& activationFunction(){
		return m_activation;
	}

	/// \brief Obtain the parameter vector.
	ParameterVectorType parameterVector() const{
		return to_vector(m_filters) | m_offset;
	}

	/// \brief Set the new parameters of the model.
	void setParameterVector(ParameterVectorType const& newParameters){
		SIZE_CHECK(newParameters.size() == numberOfParameters());
		std::size_t filterParams = m_filters.size1() * m_filters.size2();
		noalias(to_vector(m_filters)) = subrange(newParameters,0,filterParams);
		noalias(m_offset) = subrange(newParameters,filterParams,newParameters.size());
	}

	/// \brief Return the number of parameters.
	size_t numberOfParameters() const{
		return m_filters.size1() * m_filters.size2() + m_offset.size();
	}

	///\brief Sets the structure by setting the shape of image and filters.
	///
	/// \arg imageShape Shape of the image channel x imHeight x imWidth
	/// \arg filterShape Shape of the filter matrix numFilters x fiHeight x fiWidth
	/// \arg type Type of convolution padding to perform
	void setStructure(
		Shape const& imageShape, Shape const& filterShape, Padding type = Padding::ZeroPad
	){
		m_type = type;
		m_inputShape = imageShape;
		m_filterShape ={imageShape[0], filterShape[1], filterShape[2]};
		if(m_type != Padding::Valid){
			m_outputShape = {filterShape[0], imageShape[1], imageShape[2]};
		}else{
			m_outputShape = {filterShape[0], imageShape[1] - filterShape[1] + 1, imageShape[2] - filterShape[2] + 1};
		}
		m_filters.resize(filterShape[0], filterShape[1] * filterShape[2] * imageShape[0]);
		m_offset.resize(filterShape[0]);
	}

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new typename ActivationFunction::State());
	}

	using base_type::eval;

	/// Evaluate the model
	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State& state)const{
		SIZE_CHECK(inputs.size2() == m_inputShape.numElements());
		outputs.resize(inputs.size1(), m_outputShape.numElements());
		outputs.clear();
		
		//compute convolution
		//geometry for "zero pad"
		std::size_t paddingHeight = (m_type != Padding::Valid) ? m_filterShape[1] - 1: 0;
		std::size_t paddingWidth = (m_type != Padding::Valid) ? m_filterShape[2] - 1: 0;
		
		image::convolution(
			inputs, m_filters, outputs,
			m_inputShape, m_filterShape,
			paddingHeight, paddingWidth,
			ImageFormat::NCHW, ImageFormat::NCHW, false
		);
		
		//add offset
		//note: remora does not support 3d tensors, so we have to slice along the images index and add by hand
		std::size_t outputsForFilter = m_outputShape[1] * m_outputShape[2];
		for(std::size_t i = 0; i != inputs.size1(); ++i){
			auto slice = to_matrix(row(outputs, i), m_filters.size1(), outputsForFilter);
			noalias(trans(slice)) += blas::repeat(m_offset, outputsForFilter);
		}

		//compute activation
		m_activation.evalInPlace(outputs, state.toState<typename ActivationFunction::State>());
	}

	///\brief Calculates the first derivative w.r.t the parameters and sums them up over all inputs of the last computed batch
	void weightedParameterDerivative(
		BatchInputType const& inputs,
		BatchOutputType const& outputs,
		BatchOutputType const& coefficients,
		State const& state,
		ParameterVectorType& gradient
	)const{
		SIZE_CHECK(coefficients.size2()==outputShape().numElements());
		SIZE_CHECK(coefficients.size1()==inputs.size1());
		
		//backpropagate through activation
		BatchOutputType delta = coefficients;
		m_activation.multiplyDerivative(outputs,delta, state.toState<typename ActivationFunction::State>());
		
		//split gradient in offset and filter parts
		gradient.resize(numberOfParameters());
		gradient.clear();
		std::size_t filterParams = m_filters.size1() * m_filters.size2();
		auto weightGradient = to_matrix(subrange(gradient,0,filterParams), m_filters.size1(), m_filters.size2());
		auto offsetGradient = subrange(gradient, filterParams,gradient.size());
		
		//derivative of convolution wrt filters
		std::size_t paddingHeight = (m_type != Padding::Valid) ? m_filterShape[1] - 1: 0;
		std::size_t paddingWidth = (m_type != Padding::Valid) ? m_filterShape[2] - 1: 0;
		image::convolution(
			inputs, delta, weightGradient,
			m_inputShape, m_outputShape,
			paddingHeight, paddingWidth,
			ImageFormat::CNHW, ImageFormat::CNHW, false
		);
		
		
		//derivatives of offset parameters
		//note: remora does not support 3d tensors, so we have to slice along the images index and add by hand
		std::size_t outputsForFilter = m_outputShape[1] * m_outputShape[2];
		for(std::size_t i = 0; i != inputs.size1(); ++i){
			auto slice = to_matrix(row(delta, i), m_filters.size1(), outputsForFilter);
			noalias(offsetGradient) += sum(as_rows(slice));
		}
	}
	///\brief Calculates the first derivative w.r.t the inputs and summs them up over all inputs of the last computed batch
	void weightedInputDerivative(
		BatchInputType const & inputs,
		BatchOutputType const& outputs,
		BatchOutputType const & coefficients,
		State const& state,
		BatchInputType& derivatives
	)const{
		SIZE_CHECK(coefficients.size2() == outputShape().numElements());
		SIZE_CHECK(coefficients.size1() == inputs.size1());
		
		BatchOutputType delta = coefficients;
		m_activation.multiplyDerivative(outputs,delta, state.toState<typename ActivationFunction::State>());
		
		std::size_t paddingHeight = m_filterShape[1] - 1;
		std::size_t paddingWidth = m_filterShape[2] - 1;
		if(m_type == Padding::Valid){
			paddingHeight *=2;
			paddingWidth *=2;
		}
		derivatives.resize(inputs.size1(),inputShape().numElements());
		derivatives.clear();
		
		image::convolution(
			delta, m_filters, derivatives,
			m_outputShape, m_filterShape,
			paddingHeight, paddingWidth,
			ImageFormat::NCHW, ImageFormat::CNHW, true
		);
	}

	/// From ISerializable
	void read(InArchive& archive){
		archive >> m_filters;
		archive >> m_offset;
		archive >> m_inputShape;
		archive >> m_outputShape;
		archive >> m_filterShape;
		archive >> (int&) m_type;
	}
	/// From ISerializable
	void write(OutArchive& archive) const{
		archive << m_filters;
		archive << m_offset;
		archive << m_inputShape;
		archive << m_outputShape;
		archive << m_filterShape;
		archive << (int&) m_type;
	}
	
private:

	MatrixType m_filters; ///< Filters used for performing the convolution
	VectorType m_offset;///< offset applied to each filters response
	ActivationFunction m_activation;///< The activation function to use

	Shape m_inputShape; ///< shape of a single input
	Shape m_outputShape; ///< shape of a single output
	Shape m_filterShape; ///< shape of a single filter
	Padding m_type; ///> type of padding used
};


}
#endif
