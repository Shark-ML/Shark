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
#include <shark/Core/Images/Interpolation.h>
#include <shark/LinAlg/BLAS/device_copy.hpp>
namespace shark {


	
	
	

///
/// \brief Resizes an input image to a given size
///
/// \par
/// The image is resized using an interpolation algorithm which can be chosen by the user. Right now,
/// only spline interpolation is supported. This will slightly smooth input images
/// over a 4x4 grid. This also means that resizing to the same size is not an identity operation
///
/// The derivative of the model wrt its input images is available.
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
	/// \arg type Type of interpolation to perform, default is Spline-Interpolation
	ResizeLayer(
		Shape const& inputShape, Shape const& outputShape, Interpolation type = Interpolation::Spline
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
	/// \arg type Type of interpolation to perform, default is Spline-Interpolation
	void setStructure(
		Shape const& inputShape, Shape const& outputShape, Interpolation type = Interpolation::Spline
	){
		m_type = type;
		m_inputShape = inputShape;
		m_outputShape = {outputShape[0], outputShape[1], inputShape[2]};
		//compute pixel coordinates by evenly spreading them out on the image
		blas::matrix<value_type> points(outputShape[0] * outputShape[1], 2);
		for(std::size_t i = 0; i != outputShape[1]; ++i){
			for(std::size_t j = 0; j != outputShape[0]; ++j){
				points(i * outputShape[0] +j, 0) = value_type(i) / outputShape[1];
				points(i * outputShape[0] +j, 1) = value_type(j) / outputShape[0];
			}
		}
		
		m_points = copy_to_device(points, device_type());
	}

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	using base_type::eval;

	/// Evaluate the model
	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State&)const{
		SIZE_CHECK(inputs.size2() == m_inputShape.numElements());
		outputs.resize(inputs.size1(), m_outputShape.numElements());
		imageInterpolate2D<value_type, device_type>(
			inputs, m_inputShape, m_type,
			m_points, m_points.size1(),
			outputs
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
	///\brief Calculates the first derivative w.r.t the inputs and summs them up over all inputs of the last computed batch
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
		weightedImageInterpolate2DDerivative<value_type, device_type>(
			inputs, m_inputShape, m_type,
			coefficients,
			m_points, m_points.size1(),
			derivatives
		);
	}

	/// From ISerializable
	void read(InArchive& archive){
		archive >> m_points;
		archive >> (int&)m_type;
		archive >> m_inputShape;
		archive >> m_outputShape;
	}
	/// From ISerializable
	void write(OutArchive& archive) const{
		archive << m_points;
		archive << (int&)m_type;
		archive << m_inputShape;
		archive << m_outputShape;
	}
	
private:
	Interpolation m_type;
	MatrixType m_points;
	Shape m_inputShape;
	Shape m_outputShape;
};


}
#endif
