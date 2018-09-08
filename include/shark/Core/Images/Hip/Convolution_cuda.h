//===========================================================================
/*!
 * 
 *
 * \brief  Convolution of an image with a filter
 * 
 * 
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
//===========================================================================

#ifndef SHARK_CORE_IMAGES_HIP_CONVOLUTION_CUDA_H
#define SHARK_CORE_IMAGES_HIP_CONVOLUTION_CUDA_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Shark.h>
#include <shark/Core/Images/Enums.h>

#include <cudnn.h>
namespace shark{
namespace image{

template<class M1, class M2>
void convolution(
	blas::matrix_expression<M1, hip_tag> const& images, 
	blas::matrix_expression<M1, hip_tag> const& filters, 
	blas::matrix_expression<M2, hip_tag>& results,
	Shape const& shapeImage,
	Shape const& shapeFilters,
	Shape const& shapeOut,
	Padding padding
){
	SIZE_CHECK(shapeImage.size() == 3);
	SIZE_CHECK(shapeFilters.size() == 4);
	SIZE_CHECK(shapeImage[2] == shapeFilters[3] );
	SIZE_CHECK(shapeImage.numElements() == images().size2());
	SIZE_CHECK(shapeFilters[0] == filters().size1());
	SIZE_CHECK(shapeFilters[1] * shapeFilters[2] * shapeFilters[3] == filters().size2());
	
	std::size_t paddingHeight = (padding != Padding::Valid) ? shapeFilter[1] - 1: 0;
	std::size_t paddingWidth = (padding != Padding::Valid) ? shapeFilter[2] - 1: 0;
	blas::kernels::conv2d(images, filters, results,
		shapeImage[2], shapeFilters[0], 
		shapeImage[0], shapeImage[1],
		shapeFilters[1], shapeFilters[2],
		paddingHeight, paddingWidth
	);
}

template<class M, class M1, class M2, class Device>
void convolutionWeightedInputDerivative(
	blas::matrix_expression<M1, cpu_tag>&& coefficients, 
	blas::matrix_expression<M1, cpu_tag> const& filters, 
	blas::matrix_expression<M2, cpu_tag>& derivatives,
	Shape const& shapeImage,
	Shape const& shapeFilters,
	Shape const& shapeOut,
	Padding padding
){
	Shape shape = outputShape();
	std::size_t paddingHeight = m_filterHeight - 1;
	std::size_t paddingWidth = m_filterWidth - 1;
	if(m_type == Padding::Valid){
		paddingHeight *=2;
		paddingWidth *=2;
	}
	derivatives.resize(inputs.size1(),inputShape().numElements());
	derivatives.clear();
	blas::kernels::conv2d(delta, m_backpropFilters, derivatives,
		shapeImage[2], shapeFilters[0], 
		shapeOut[0], shapeOut[1],
		m_filterHeight, m_filterWidth,
		paddingHeight, paddingWidth
	);
	
	//~ typedef typename M1::value_type value_type;
	//~ //reorder to CHWN
	//~ std::size_t n = coefficients.size1();
	//~ BatchOutputType coeffs_CHWN(shapeFilters[0], coefficients.size1() * coefficients.size2() / shapeFilters[0]);
	//~ BatchOutputType inputs_CHWN(shapeImage[2],shapeImage[0] * shapeImage[1] * n);
	//~ image::reorder<value_type, device_type>(
		//~ to_vector(delta), to_vector(delta_CHWN), 
		//~ {n, outputHeight, outputWidth, m_numFilters},
		//~ ImageFormat::NHWC, ImageFormat::CHWN
	//~ );
	//~ image::reorder<value_type, device_type>(
		//~ to_vector(inputs), to_vector(inputs_CHWN),
		//~ {n, m_imageHeight, m_imageWidth, m_numChannels},
		//~ ImageFormat::NHWC, ImageFormat::CHWN
	//~ );
	//~ BatchInputType responses_CHWN(m_numChannels, m_filters.size() / m_numChannels);
	//~ blas::kernels::conv2d(inputs_CHWN, to_vector(delta_CHWN), responses_CHWN,
		//~ n, m_numFilters, 
		//~ m_imageHeight, m_imageWidth,
		//~ outputHeight, outputWidth,
		//~ paddingHeight, paddingWidth
	//~ );
	//~ image::reorder<value_type, device_type>(
		//~ to_vector(responses_CHWN), weightGradient, 
		//~ {m_numChannels, m_filterHeight, m_filterWidth, m_numFilters}, 
		//~ ImageFormat::CHWN, ImageFormat::NHWC
	//~ );
}


}}

#endif