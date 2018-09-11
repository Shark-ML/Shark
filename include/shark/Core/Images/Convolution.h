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

#ifndef SHARK_CORE_IMAGES_CONV_H
#define SHARK_CORE_IMAGES_CONV_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Shark.h>

//implementations 
#include <shark/Core/Images/CPU/Convolution.h>

#ifdef __NVCC__
#include <shark/Core/Images/Hip/ConvolutionCuda.h>
#endif

//~ namespace shark{namespace image{

//~ /// \brief Computes the convolution between a given set of images and each provided filter
//~ ///
//~ /// The exact format of images and filters depends on the provided format. Only NCHW is supported on all computing formats
//~ /// If format is set to NCHW, the rows of the matrix store one image each in CHW format. 
//~ /// In this case, the shape must provide the CHW format description.
//~ /// If format is set to CNHW, each row stores one channel. thus the columns are storing several images in NHW format. 
//~ /// In this case, the shape must provide the NHW format description. This is only supported on CPU.
//~ ///
//~ /// outputs store the data in the same format as the imageFormat. The number of channels of the output
//~ /// is the same as the number of filters used. Output must be correctly sized to hold the output. The dimensions are
//~ /// output.width = image.width -filter.width +1 + paddingWidth
//~ /// output.height = image.height -filter.height +1 + paddingHeight
//~ ///
//~ /// Padding is done using zero-padding.
//~ /// The pad is split equally around the borders, where the top and left border width are rounded down.
//~ /// The filter can be flipped (flip=true creates a true convolution, while false gives a cross-correlation). The padding
//~ /// is taking the flip into account (flip -> top and left is rounded up)
//~ ///
//~ /// \param images The images to convolve, format given by imageFormat
//~ /// \param filters The filters to use, format given by filterFormat
//~ /// \param results Stores the result, format given by imageFormat
//~ /// \param shapeImages shape of the minor 3 image dimensions CHW or NHW depending on imageFormat
//~ /// \param shapeFilters shape of the minor 3 filter dimensions CHW or NHW depending on filterFormat
//~ /// \param paddingHeight the padding applied to the rows
//~ /// \param paddingWidth the padding applied to the columns
//~ /// \param flipFilters whether to flip the H and W axes of the filter
//~ template<class M1, class M2, class Device>
//~ void convolution(
	//~ blas::matrix_expression<M1, Device> const& images, 
	//~ blas::matrix_expression<M1, Device> const& filters, 
	//~ blas::matrix_expression<M2, Device>& results,
	//~ Shape const& shapeImages,
	//~ Shape const& shapeFilters,
	//~ std::size_t paddingHeight,
	//~ std::size_t paddingWidth,
	//~ ImageFormat imageFormat,
	//~ ImageFormat filterFormat,
	//~ bool flipFilters
//~ ){
	//~ detail::convolution(
		//~ images, filters, results, 
		//~ shapeImages, shapeFilters,
		//~ paddingHeight, paddingWidth,
		//~ imageFormat, filterFormat,
		//~ flipFilters
	//~ );
//~ }
	
//~ }}



#endif