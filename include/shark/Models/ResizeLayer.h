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
#ifndef SHARK_MODELS_RESIZE_LAYER_H
#define SHARK_MODELS_RESIZE_LAYER_H

#include <shark/Models/AbstractModel.h>
#include <shark/Core/Images/Resize.h>
#include <shark/LinAlg/BLAS/device_copy.hpp>
namespace shark {
///
/// \brief Resizes an input image to a given size
///
/// \par
/// The image is resized using an interpolation algorithm which can be chosen by the user. 
/// In general, resizing with the same size might not be an identity operation, e.g. performing
/// a spline-interpolation will smooth the image.
/// Right now, only linear interpolation is supported.
///
/// Implementation details: 
/// Interpolation is implemented such that corners are aligned and the area
/// of each pixel is mapped roughly to a same-size area in the output image. This means that
/// when upsampling, the border of the image needs to be padded as each pixel is replaced
/// by a set of pixels in the same area. We use zero-padding,
/// which might lead to noticable artefacts (dark border) in small images.
///
/// Note that scaling down by a factor larger than two is
/// not a good idea with most interpolation schemes as this can lead to ringing and other artifacts.
///
/// \ingroup models
template <class VectorType = RealVector>
class ResizeLayer : public AbstractModel<VectorType,VectorType,VectorType>{
private:
	typedef typename VectorType::value_type value_type;
	typedef typename VectorType::device_type device_type;
	typedef blas::matrix<value_type, blas::row_major, device_type> MatrixType;
	typedef AbstractModel<VectorType,VectorType,VectorType> base_type;
	static_assert(!std::is_same<typename VectorType::storage_type::storage_tag, blas::dense_tag>::value, "Resizing not supported for sparse inputs");
public:
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ParameterVectorType ParameterVectorType;

	/// Default Constructor; use setStructure later.
	ResizeLayer(){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}
	
	///\brief Configures the model.
	///
	/// \arg inputShape Shape of the image imHeight x imWidth x channel
	/// \arg outputShape Shape of the resized output imHeight x imWidth
	/// \arg type Type of interpolation to perform, default is Linear-Interpolation
	ResizeLayer(
		Shape const& inputShape, Shape const& outputShape, Interpolation type = Interpolation::Linear
	){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		setStructure(inputShape, outputShape, type);
	}

	std::string name() const
	{ return "ResizeLayer"; }
	
	///\brief Returns the expected shape of the input
	Shape inputShape() const{
		return m_inputShape;
	}
	///\brief Returns the shape of the output
	Shape outputShape() const{
		return m_outputShape;
	}
	
	/// \brief Obtain the parameter vector.
	ParameterVectorType parameterVector() const{
		return ParameterVectorType();
	}

	/// \brief Set the new parameters of the model.
	void setParameterVector(ParameterVectorType const& newParameters){}

	/// \brief Return the number of parameters.
	size_t numberOfParameters() const{
		return 0;
	}

	///\brief Configures the model.
	///
	/// \arg inputShape Shape of the image imHeight x imWidth x channel
	/// \arg outputShape Shape of the resized output imHeight x imWidth
	/// \arg type Type of interpolation to perform, default is Linear-Interpolation
	void setStructure(
		Shape const& inputShape, Shape const& outputShape, Interpolation type = Interpolation::Linear
	){
		SHARK_RUNTIME_CHECK(type == Interpolation::Linear, "Sorry, only linear interpolation is currently supported");
		m_type = type;
		m_inputShape = inputShape;
		m_outputShape = {outputShape[0], outputShape[1], inputShape[2]};
	}

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	using base_type::eval;

	/// Evaluate the model
	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State&)const{
		SIZE_CHECK(inputs.size2() == m_inputShape.numElements());
		outputs.resize(inputs.size1(), m_outputShape.numElements());
		image::resize(
			inputs, outputs, 
			m_inputShape, m_outputShape,
			m_type
		);
	}

	///\brief Calculates the first derivative w.r.t the parameters and summing them up over all inputs of the last computed batch
	void weightedParameterDerivative(
		BatchInputType const& inputs,
		BatchOutputType const& outputs,
		BatchOutputType const& coefficients,
		State const& state,
		ParameterVectorType& gradient
	)const{}
	///\brief Calculates the first derivative w.r.t the inputs and sums them up over all inputs of the last computed batch
	void weightedInputDerivative(
		BatchInputType const & inputs,
		BatchOutputType const& outputs,
		BatchOutputType const & coefficients,
		State const& state,
		BatchInputType& derivatives
	)const{
		SIZE_CHECK(inputs.size2() == m_inputShape.numElements());
		SIZE_CHECK(outputs.size2() == m_outputShape.numElements());
		SIZE_CHECK(coefficients.size2() == outputs.size2());
		SIZE_CHECK(coefficients.size1() == inputs.size1());
		SIZE_CHECK(outputs.size1() == inputs.size1());
		
		derivatives.resize(inputs.size1(), m_inputShape.numElements());
		image::resizeWeightedDerivative(
			inputs, coefficients, derivatives, 
			m_inputShape, m_outputShape,
			m_type
		);
	}

	/// From ISerializable
	void read(InArchive& archive){
		archive >> (int&)m_type;
		archive >> m_inputShape;
		archive >> m_outputShape;
	}
	/// From ISerializable
	void write(OutArchive& archive) const{
		archive << (int&)m_type;
		archive << m_inputShape;
		archive << m_outputShape;
	}
	
private:
	Interpolation m_type;
	Shape m_inputShape;
	Shape m_outputShape;
};


}
#endif
