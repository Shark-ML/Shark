#ifndef SHARK_CORE_IMAGES_INTERPOLATION_H
#define SHARK_CORE_IMAGES_INTERPOLATION_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Shark.h>
#include "CPU/SplineInterpolation2D.h"
#ifdef SHARK_USE_OPENCL
#include "OpenCL/SplineInterpolation2D.h"
#endif
namespace shark{
enum class Interpolation{
	Spline
};

/// \brief Performs interpolation of an image at a set of evaluation points.
///
/// Uses the chosen interpolation method to compute the interpolated values at a given point (y,x)
/// (y,x)-coordinates are encoded such that (0,0) maps to the (0,0) pixel and (1,1) maps to (height, width )
/// in the image. 
///
/// The method can interpolate a batch of images at the same time.
/// For batch-processing, either all images are interpolated with the same points, 
/// in which case points.size1() = pointsPerImage or each image uses its own set of points.
template<class T, class Device>
void imageInterpolate2D(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, Device> images, 
	Shape const& shape,
	Interpolation method,
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, Device> points,
	std::size_t pointsPerImage, 
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, Device> values
){
	if(method == Interpolation::Spline){
		image::splineInterpolation2D(images, shape, points, pointsPerImage, values);
	}
}

/// \brief Performs interpolation of an image at a set of evaluation points. as well as the derivative wrt the point coordinates.
///
/// Uses the chosen interpolation method to compute the interpolated values at a given point (y,x)
/// (x,x)-coordinates are encoded such that (0,0) maps to the (0,0) pixel and (1,1) maps to (height, width)
/// in the image. 
///
/// The method can interpolate a batch of images at the same time.
/// For batch-processing, either all images are interpolated with the same points, 
/// in which case points.size1() = pointsPerImage or each image uses its own set of points.
template<class T, class Device>
void imageInterpolate2D(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, Device> images, 
	Shape const& shape,
	Interpolation method,
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, Device> points,
	std::size_t pointsPerImage, 
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, Device> values,
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, Device> valuesdx,
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, Device> valuesdy
){
	if(method == Interpolation::Spline){
		image::splineInterpolation2D(images, shape, points, pointsPerImage, values, valuesdx, valuesdy);
	}
}

/// \brief Computes the chain role of the derivative of the chosen interpolation method wrt the interpolated image
///
/// This method does n
template<class T, class Device>
void weightedImageInterpolate2DDerivative(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, Device> images, 
	Shape const& shape,
	Interpolation method,
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, Device> imageDerivatives, 
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, Device> points,
	std::size_t pointsPerImage, 
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, Device> derivatives
){
	if(method == Interpolation::Spline){
		image::splineInterpolation2DDerivative(imageDerivatives, shape, points, pointsPerImage, derivatives);
	}
	(void) images;
}
}

#endif