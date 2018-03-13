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
#include <shark/Models/ConvolutionalModel.h>
 
namespace shark{

template <class VectorType = RealVector>
class PoolingLayer : public AbstractModel<VectorType, VectorType, VectorType>{
private:
	typedef AbstractModel<VectorType,VectorType, VectorType> base_type;
	Shape m_inputShape;
	Shape m_outputShape;
	Shape m_patch;
	Padding m_padding;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::ParameterVectorType ParameterVectorType;

	PoolingLayer(Shape const& inputShape, Shape const& patchShape, Padding padding)
	: m_inputShape(inputShape), m_patch(patchShape), m_padding(padding){
		base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		base_type::m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		
		if(m_padding == Padding::Valid)
			m_outputShape =  {m_inputShape[0]/m_patch[0], m_inputShape[1]/m_patch[1], m_inputShape[2]};
		else
			m_outputShape = {
				(m_inputShape[0] + m_patch[0] - 1)/m_patch[0], 
				(m_inputShape[1] + m_patch[1] - 1)/m_patch[1], 
				m_inputShape[2]
			};
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

	using base_type::eval;

	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State& state)const{
		SIZE_CHECK(inputs.size2() == m_inputShape.numElements());
		outputs.resize(inputs.size1(),m_outputShape.numElements());
		
		//for all images
		for(std::size_t img = 0; img != inputs.size1(); ++img){
			//Extract single images and create matrices (pixels,channels)
			auto imageIn = to_matrix(row(inputs,img), m_inputShape[0] * m_inputShape[1], m_inputShape[2]);
			auto imageOut = to_matrix(row(outputs,img), m_outputShape[0] * m_outputShape[1], m_outputShape[2]);
			//traverse over all pixels of the output image
			for(std::size_t p = 0; p != imageOut.size1(); ++p){
				auto pixel = row(imageOut,p);
				//extract pixel coordinates in input image
				std::size_t starti = (p / m_outputShape[1]) * m_patch[0];
				std::size_t startj = (p % m_outputShape[1]) * m_patch[1];
				
				//traverse the patch on the input image and compute maximum
				noalias(pixel) = row(imageIn, starti * m_inputShape[1] + startj);
				for(std::size_t i = starti; i != starti + m_patch[0]; ++i){
					for(std::size_t j = startj; j != startj + m_patch[1]; ++j){
						std::size_t index = i * m_inputShape[1] + j;
						noalias(pixel) = max(pixel,row(imageIn, index));
					}
				}
			}
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
		SIZE_CHECK(coefficients.size1() == outputs.size1());
		SIZE_CHECK(coefficients.size2() == outputs.size2());
		std::size_t outputPixels = m_outputShape[0] * m_outputShape[1];
		derivative.resize(inputs.size1(),inputs.size2());
		derivative.clear();
		
		//for all images
		for(std::size_t img = 0; img != inputs.size1(); ++img){
			//Extract single images and create matrices (pixels,channels)
			auto imageCoeffs = to_matrix(row(coefficients,img), m_outputShape[0] * m_outputShape[1], m_outputShape[2]);
			auto imageIn = to_matrix(row(inputs,img), m_inputShape[0] * m_inputShape[1], m_inputShape[2]);
			auto imageDer = to_matrix(row(derivative,img), m_inputShape[0] * m_inputShape[1], m_inputShape[2]);
			//traverse over all pixels of the output image
			for(std::size_t p = 0; p != outputPixels; ++p){
				//extract pixel coordinates in input image
				std::size_t starti = (p / m_outputShape[1]) * m_patch[0];
				std::size_t startj = (p % m_outputShape[1]) * m_patch[1];
				//traverse the patch on the input image and compute arg-max for each channel
				for(std::size_t c = 0; c != m_inputShape[2]; ++c){
					std::size_t maxIndex =  starti * m_inputShape[1] + startj;
					double maxVal = imageIn(maxIndex,c);
					for(std::size_t i = starti; i != starti + m_patch[0]; ++i){
						for(std::size_t j = startj; j != startj + m_patch[1]; ++j){
							std::size_t index = i * m_inputShape[1] + j;
							double val = imageIn(index,c);
							if(val > maxVal){
								maxVal = val;
								maxIndex = index;
							}
						}
					}
					//after arg-max is obtained, update gradient
					imageDer(maxIndex, c) += imageCoeffs(p,c);
				}
			}
		}
		
	}

	/// From ISerializable
	void read(InArchive& archive){}
	/// From ISerializable
	void write(OutArchive& archive) const{}
};


}

#endif
