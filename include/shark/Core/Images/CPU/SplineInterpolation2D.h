#ifndef SHARK_CORE_IMAGES_CPU_SPLINE_INTERPOLATION_2D_H
#define SHARK_CORE_IMAGES_CPU_SPLINE_INTERPOLATION_2D_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Shape.h>
#include <shark/Core/OpenMP.h>
namespace shark{
namespace image{
template<class T>
void splineInterpolation2D(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> images, 
	Shape const& shape,
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> points, std::size_t pointsPerImage, 
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> values
){
	std::size_t height = shape[0];
	std::size_t width = shape[1];
	std::size_t numChannels = shape[2];
	std::size_t stride = (pointsPerImage == points.size1())? 0: pointsPerImage;
	
	values.clear();
	
	SHARK_PARALLEL_FOR(int im = 0; im < (int)images.size1(); ++im){
		auto image = to_matrix(row(images,im), width * height, numChannels);
		auto v = to_matrix(row(values,im), pointsPerImage, numChannels);
		for(std::size_t p = 0; p != pointsPerImage; ++p){
			std::size_t pointIdx = p + im * stride;
			using std::min; using std::max;
			T basex = std::floor(points(pointIdx,1) * width);
			T t=points(pointIdx, 1) * width - basex;
			T t2=t*t;
			T t3=t2*t;

			T x[4]={-t3+3*t2-3*t+1, 3*t3-6*t2+4, -3*t3+3*t2+3*t+1,  t3};
			std::size_t px[4]={
				(std::size_t)min<T>(max<T>(basex-1,0),width-1),
				(std::size_t)max<T>(min<T>(basex,width-1),0),
				(std::size_t)min<T>(max<T>(basex+1,0),width-1),
				(std::size_t) max<T>(min<T>(basex+2,width-1),0)
			};
			
			T basey = std::floor(points(pointIdx,0) * height);
			t=points(pointIdx, 0) * height - basey;
			t2=t*t;
			t3=t2*t;

			T y[4]={-t3+3*t2-3*t+1, 3*t3-6*t2+4, -3*t3+3*t2+3*t+1,  t3};
			std::size_t py[4]={
				(std::size_t)min<T>(max<T>(basey-1,0),height-1),
				(std::size_t)max<T>(min<T>(basey,height-1),0),
				(std::size_t)min<T>(max<T>(basey+1,0),height-1),
				(std::size_t) max<T>(min<T>(basey+2,height-1),0)
			};
			for(std::size_t k = 0;k < 4; k++){
				for(std::size_t l = 0;l < 4; l++){
					std::size_t index = px[l] + width * py[k];
					noalias(row(v,p)) += (x[l]*y[k]/36) * row(image,index);
				}
			}
		}
	}
}

template<class T>
void splineInterpolation2D(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> images, 
	Shape const& shape,
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> points, std::size_t pointsPerImage, 
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> values,
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> valuesdx,
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> valuesdy
){
	std::size_t height = shape[0];
	std::size_t width = shape[1];
	std::size_t numChannels = shape[2];
	std::size_t stride = (pointsPerImage == points.size1())? 0: pointsPerImage;
	
	values.clear();
	valuesdx.clear();
	valuesdy.clear();
	SHARK_PARALLEL_FOR(int im = 0; im < (int)images.size1(); ++im){
		auto image = to_matrix(row(images,im), width * height, numChannels);
		auto v = to_matrix(row(values,im), pointsPerImage, numChannels);
		auto vdx = to_matrix(row(valuesdx,im), pointsPerImage, numChannels);
		auto vdy = to_matrix(row(valuesdy,im), pointsPerImage, numChannels);
		for(std::size_t p = 0; p != points.size1(); ++p){
			std::size_t pointIdx = p + im * stride;
			using std::min; using std::max;
			T basex = std::floor(points(pointIdx,1) * width);
			T t=points(pointIdx, 1) * width - basex;
			T t2=t*t;
			T t3=t2*t;

			T x[4]={-t3+3*t2-3*t+1, 3*t3-6*t2+4, -3*t3+3*t2+3*t+1,  t3};
			T dxdt[4]={-3*t2+6*t-3, 9*t2-12*t, -9*t2+6*t+3,  3*t2};
			std::size_t px[4]={
				(std::size_t)min<T>(max<T>(basex-1,0),width-1),
				(std::size_t)max<T>(min<T>(basex,width-1),0),
				(std::size_t)min<T>(max<T>(basex+1,0),width-1),
				(std::size_t) max<T>(min<T>(basex+2,width-1),0)
			};
			
			T basey = std::floor(points(pointIdx,0) * height);
			t=points(pointIdx, 0) * height - basey;
			t2=t*t;
			t3=t2*t;

			T y[4]={-t3+3*t2-3*t+1, 3*t3-6*t2+4, -3*t3+3*t2+3*t+1,  t3};
			T dydt[4]={-3*t2+6*t-3, 9*t2-12*t, -9*t2+6*t+3,  3*t2};
			std::size_t py[4]={
				(std::size_t)min<T>(max<T>(basey-1,0),height-1),
				(std::size_t)max<T>(min<T>(basey,height-1),0),
				(std::size_t)min<T>(max<T>(basey+1,0),height-1),
				(std::size_t) max<T>(min<T>(basey+2,height-1),0)
			};
			
			for(std::size_t k = 0;k < 4; k++){
				for(std::size_t l = 0;l < 4; l++){
					std::size_t index = px[l] + width * py[k];
					T vdI =x[l]*y[k]/36;
					T dfdx1 = dxdt[l] * y[k]/36;
					T dfdx2 = x[l] * dydt[k]/36;
					
					noalias(row(v,p)) += vdI * row(image,index);
					noalias(row(vdx,p)) += dfdx1 * row(image,index);
					noalias(row(vdy,p)) += dfdx2 * row(image,index);
				}
			}
		}
	}
}
template<class T>
void splineInterpolation2DDerivative(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> imageDerivatives, 
	Shape const& shape,
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> points, std::size_t pointsPerImage, 
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> results
){
	std::size_t height = shape[0];
	std::size_t width = shape[1];
	std::size_t numChannels = shape[2];
	std::size_t stride = (pointsPerImage == points.size1())? 0: pointsPerImage;
	
	results.clear();
	
	SHARK_PARALLEL_FOR(int im = 0; im < (int)imageDerivatives.size1(); ++im){
		auto imageDer = to_matrix(row(imageDerivatives,im), pointsPerImage, numChannels);
		auto result = to_matrix(row(results,im), width * height, numChannels);
		for(std::size_t p = 0; p != points.size1(); ++p){
			std::size_t pointIdx = p + im * stride;
			using std::min; using std::max;
			T basex = std::floor(points(pointIdx,1) * width);
			T t=points(pointIdx, 1) * width - basex;
			T t2=t*t;
			T t3=t2*t;

			T x[4]={-t3+3*t2-3*t+1, 3*t3-6*t2+4, -3*t3+3*t2+3*t+1,  t3};
			std::size_t px[4]={
				(std::size_t)min<T>(max<T>(basex-1,0),width-1),
				(std::size_t)max<T>(min<T>(basex,width-1),0),
				(std::size_t)min<T>(max<T>(basex+1,0),width-1),
				(std::size_t) max<T>(min<T>(basex+2,width-1),0)
			};
			
			T basey = std::floor(points(pointIdx,0) * height);
			t=points(pointIdx, 0) * height - basey;
			t2=t*t;
			t3=t2*t;

			T y[4]={-t3+3*t2-3*t+1, 3*t3-6*t2+4, -3*t3+3*t2+3*t+1,  t3};
			std::size_t py[4]={
				(std::size_t)min<T>(max<T>(basey-1,0),height-1),
				(std::size_t)max<T>(min<T>(basey,height-1),0),
				(std::size_t)min<T>(max<T>(basey+1,0),height-1),
				(std::size_t) max<T>(min<T>(basey+2,height-1),0)
			};
			for(std::size_t k = 0;k < 4; k++){
				for(std::size_t l = 0;l < 4; l++){
					std::size_t index = px[l] + width * py[k];
					T vdI = x[l]*y[k]/36;
					noalias(row(result,index)) += vdI * row(imageDer,p);
				}
			}
		}
	}
}

}}

#endif