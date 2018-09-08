//===========================================================================
/*!
 * 
 *
 * \brief       Resize images
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

#ifndef SHARK_CORE_IMAGES_RESIZE_H
#define SHARK_CORE_IMAGES_RESIZE_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Shark.h>
#include <shark/Core/Images/Enums.h>
#include "CPU/BilinearResize.h"
#ifdef SHARK_USE_OPENCL
#include "OpenCL/SplineInterpolation2D.h"
#endif
namespace shark{
namespace image{
/// \brief Resizes an image to a new giving size using an appropriate interpolation method
///
/// The method can interpolate a batch of images at the same time.
/// Currently, only linear interpolation is supported.
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
template<class M1, class M2, class Device>
void resize(
	blas::matrix_expression<M1, Device> const& images, 
	blas::matrix_expression<M2, Device>& resizedImages, 
	Shape const& shapeIn,
	Shape const& shapeOut,
	Interpolation method = Interpolation::Linear
){
	SIZE_CHECK(shapeIn.size() == shapeOut.size());
	SIZE_CHECK(shapeIn.size() <= 3);
	SIZE_CHECK(shapeIn.size() == 2 || shapeIn[0] == shapeOut[0] );
	if(method == Interpolation::Linear){
		image::bilinearResize(
			typename M1::const_closure_type(images()),
			typename M2::closure_type(resizedImages()),
			shapeIn, shapeOut
		);
	}else{
		throw SHARKEXCEPTION("Unsupported interpolation");
	}
}

/// \brief Derivative of resize with respect to the input images
///
/// The method can interpolate a batch of images at the same time.
/// Currently, only linear interpolation is supported.
template<class M, class M1, class M2, class Device>
void resizeWeightedDerivative(
	blas::matrix_expression<M, Device> const& images,
	blas::matrix_expression<M1, Device> const& coefficients, 
	blas::matrix_expression<M2, Device>& inputDerivatives, 
	Shape const& shapeIn,
	Shape const& shapeOut,
	Interpolation method = Interpolation::Linear
){
	SIZE_CHECK(images().size1() == coefficients().size1());
	SIZE_CHECK(images().size1() == inputDerivatives().size1());
	SIZE_CHECK(images().size2() == inputDerivatives().size2());
	SIZE_CHECK(shapeIn.size() == shapeOut.size());
	SIZE_CHECK(shapeIn.size() <= 3);
	SIZE_CHECK(shapeIn.size() == 2 || shapeIn[0] == shapeOut[0] );
	if(method == Interpolation::Linear){
		image::bilinearResizeWeightedDerivative(
			typename M1::const_closure_type(coefficients()),
			typename M2::closure_type(inputDerivatives()),
			shapeIn, shapeOut);
	}else{
		throw SHARKEXCEPTION("Unsupported interpolation");
	}
	(void)images;//prevent warning
}



template<class M, class Device>
blas::matrix<typename M::value_type, blas::row_major, Device> resize(
	blas::matrix_expression<M, Device> const& images, 
	Shape const& shapeIn,
	Shape const& shapeOut,
	Interpolation method = Interpolation::Linear
){
	blas::matrix<typename M::value_type, blas::row_major, Device> resizedImages(images().size1(), shapeOut.numElements());
	resize(images(), resizedImages, shapeIn, shapeOut, method);
	return resizedImages;
}


template<class V, class Device>
blas::vector<typename V::value_type, Device> resize(
	blas::vector_expression<V, Device> const& images, 
	Shape const& shapeIn,
	Shape const& shapeOut,
	Interpolation method = Interpolation::Linear
){
	blas::vector<typename V::value_type, Device> resized(shapeOut.numElements());
	auto matResult = to_matrix(resized, 1, resized.size());
	resize(
		to_matrix(images,1, images().size()),
		matResult,
		shapeIn, shapeOut, method
	);
	return resized;
}

}}

#endif