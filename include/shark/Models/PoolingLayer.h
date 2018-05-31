/*!
 * 
 *
 * \brief       Creates pooling layers
 *
 * \author      O.Krause
 * \date        2018
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
#ifndef SHARK_MODELS_POOLING_LAYER_H
#define SHARK_MODELS_POOLING_LAYER_H
 
#include <shark/LinAlg/Base.h>
#include <shark/Models/AbstractModel.h>
#include <shark/Core/Images/Padding.h>
#include <shark/Core/Images/CPU/Pooling.h>
#ifdef SHARK_USE_OPENCL
#include <shark/Core/Images/OpenCL/Pooling.h>
 #endif
 
namespace shark{
	
enum class Pooling{
	Maximum
};

/// \brief Performs Pooling operations for a given input image.
///
/// Pooling partitions the input images in rectangular regions, typically 2x2 and computes 
/// a statistic over the data. This could for example be the maximum or average of values. This is
/// done channel-by-channel. The output is a smaller image where each pixel includes
/// for each channel the computed statistic. Therefore, if the patch is 2x2 the output image will have half width and height.
///
/// \ingroup models
template <class VectorType = RealVector>
class PoolingLayer : public AbstractModel<VectorType, VectorType, VectorType>{
private:
	typedef AbstractModel<VectorType,VectorType, VectorType> base_type;
	typedef typename VectorType::value_type value_type;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::ParameterVectorType ParameterVectorType;

	PoolingLayer(){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}
	
	PoolingLayer(Shape const& inputShape, Shape const& patchShape, Pooling pooling = Pooling::Maximum, Padding padding = Padding::Valid){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		setStructure(inputShape, patchShape, pooling, padding);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NeuronLayer"; }
	
	Shape inputShape() const{
		return m_inputShape;
	}
	
	Shape outputShape() const{
		return m_outputShape;
	}

	/// obtain the parameter vector
	ParameterVectorType parameterVector() const{
		return ParameterVectorType();
	}

	/// overwrite the parameter vector
	void setParameterVector(ParameterVectorType const& newParameters){
		SIZE_CHECK(newParameters.size() == 0);
	}

	/// returns the number of parameters
	size_t numberOfParameters() const{
		return 0;
	}

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}
	
	///\brief Configures the model.
	///
	/// \arg inputShape Shape of the image imHeight x imWidth x channel
	/// \arg outputShape Shape of the resized output imHeight x imWidth
	/// \arg type Type of interpolation to perform, default is Spline-Interpolation
	void setStructure(
		Shape const& inputShape, Shape const& patchShape, Pooling type = Pooling::Maximum, Padding padding = Padding::Valid
	){
		SHARK_RUNTIME_CHECK( padding == Padding::Valid, "Padding not implemented");
		m_inputShape = inputShape;
		m_patch = patchShape;
		m_padding = padding;
		m_type = type;
		if(m_padding == Padding::Valid)
			m_outputShape =  {m_inputShape[0]/m_patch[0], m_inputShape[1]/m_patch[1], m_inputShape[2]};
		else
			m_outputShape = {
				(m_inputShape[0] + m_patch[0] - 1)/m_patch[0], 
				(m_inputShape[1] + m_patch[1] - 1)/m_patch[1], 
				m_inputShape[2]
			};
	}

	using base_type::eval;

	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State& state)const{
		SIZE_CHECK(inputs.size2() == m_inputShape.numElements());
		outputs.resize(inputs.size1(),m_outputShape.numElements());
		switch(m_type){
			case Pooling::Maximum:
				image::maxPooling<value_type>(inputs, m_inputShape, m_patch, outputs);
			break;
		}
	}

	///\brief Calculates the first derivative w.r.t the parameters and summing them up over all inputs of the last computed batch
	void weightedParameterDerivative(
		BatchInputType const& inputs, 
		BatchOutputType const& outputs,
		BatchOutputType const& coefficients,
		State const& state, 
		ParameterVectorType& gradient
	)const{
		SIZE_CHECK(coefficients.size1()==outputs.size1());
		SIZE_CHECK(coefficients.size2()==outputs.size2());
		gradient.resize(0);
	}
	///\brief Calculates the first derivative w.r.t the inputs and summs them up over all inputs of the last computed batch
	void weightedInputDerivative(
		BatchInputType const & inputs,
		BatchOutputType const & outputs,
		BatchOutputType const & coefficients,
		State const& state,
		BatchInputType& derivative
	)const{
		SIZE_CHECK(coefficients.size1() == outputs.size1());
		SIZE_CHECK(coefficients.size2() == outputs.size2());
		derivative.resize(inputs.size1(),inputs.size2());
		switch(m_type){
			case Pooling::Maximum:
				image::maxPoolingDerivative<value_type>(inputs, coefficients, m_inputShape, m_patch, derivative);
			break;
		}
	}

	/// From ISerializable
	void read(InArchive& archive){
		archive >> m_inputShape;
		archive >> m_outputShape;
		archive >> m_patch;
		archive >> (int&)m_padding;
		archive >> (int&)m_type;
	}
	/// From ISerializable
	void write(OutArchive& archive) const{
		archive << m_inputShape;
		archive << m_outputShape;
		archive << m_patch;
		archive << (int&)m_padding;
		archive << (int&)m_type;
	}
private:
	Shape m_inputShape;
	Shape m_outputShape;
	Shape m_patch;
	Padding m_padding;
	Pooling m_type;
};


}

#endif
