#ifndef SHARK_CORE_IMAGES_CPU_POOLING_2D_H
#define SHARK_CORE_IMAGES_CPU_POOLING_2D_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Exception.h>
#include <shark/Core/Shape.h>
namespace shark{
namespace image{
template<class T>
void maxPooling(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> inputs, 
	Shape const& shape,
	Shape const& patchSize,
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> outputs
){
	std::size_t outputHeight = shape[1]/patchSize[0];
	std::size_t outputWidth = shape[2]/patchSize[1];
	std::size_t outputPixels = outputWidth * outputHeight;
	SIZE_CHECK(outputs.size2() == outputPixels * shape[0]);
	
	//reshape to giant single image with shape[0]*numImages channels which are processed independently.
	auto channelsIn = to_matrix(to_vector(inputs), shape[0] * inputs.size1(), shape[1] * shape[2]);
	auto channelsOut = to_matrix(to_vector(outputs), shape[0] * inputs.size1(), outputHeight * outputWidth);
	//for all images and channels
	for(std::size_t img = 0; img != inputs.size1(); ++img){
		for(std::size_t c = 0; c != shape[0]; ++c){
			//Extract current channel
			auto imageIn = to_matrix(row(channelsIn,img * shape[0] + c), shape[1], shape[2]);
			auto imageOut = to_matrix(row(channelsOut,img * shape[0] + c), outputHeight, outputWidth);
			//traverse over all pixels of the output image
			for(std::size_t i = 0; i != outputHeight; ++i){
				for(std::size_t j = 0; j != outputWidth; ++j){
					T& pixel = imageOut(i,j);
					//extract pixel coordinates in input image
					std::size_t starti = i * patchSize[0];
					std::size_t startj = j * patchSize[1];
					std::size_t endi = starti + patchSize[0];
					std::size_t endj = startj + patchSize[1];
					
					//traverse the patch on the input image and compute maximum
					pixel = imageIn(starti,startj);
					for(std::size_t i0 = starti; i0 != endi; ++i0){
						for(std::size_t j0 = startj; j0 != endj; ++j0){
							pixel = std::max(pixel, imageIn(i0,j0));
						}
					}
				}
			}
		}
	}
}

template<class T>
void maxPoolingDerivative(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> inputs, 
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> coefficients, 
	Shape const& shape,
	Shape const& patchSize,
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> derivatives
){
	std::size_t outputHeight = shape[1]/patchSize[0];
	std::size_t outputWidth = shape[2]/patchSize[1];
	std::size_t outputPixels = outputWidth * outputHeight;
	SIZE_CHECK(coefficients.size2() == outputPixels * shape[0]);
	derivatives.clear();
	
	//reshape to giant single image with shape[0]*numImages channels which are processed independently.
	auto channelsIn = to_matrix(to_vector(inputs), shape[0] * inputs.size1(), shape[1] * shape[2]);
	auto channelsCoeffs = to_matrix(to_vector(coefficients), shape[0] * inputs.size1(), outputHeight * outputWidth);
	auto channelsDer = to_matrix(to_vector(derivatives), shape[0] * inputs.size1(), shape[1] * shape[2]);
	//for all images and channels
	for(std::size_t img = 0; img != inputs.size1(); ++img){
		for(std::size_t c = 0; c != shape[0]; ++c){
			//Extract current channel
			auto imageCoeffs = to_matrix(row(channelsCoeffs,img * shape[0] + c), outputHeight, outputWidth);
			auto imageIn = to_matrix(row(channelsIn,img * shape[0] + c), shape[1], shape[2]);
			auto imageDer = to_matrix(row(channelsDer,img * shape[0] + c), shape[1], shape[2]);
			//traverse over all pixels of the output image
			for(std::size_t i = 0; i != outputHeight; ++i){
				for(std::size_t j = 0; j != outputWidth; ++j){
					//extract pixel coordinates in input image
					std::size_t starti = i * patchSize[0];
					std::size_t startj = j * patchSize[1];
					std::size_t endi = starti + patchSize[0];
					std::size_t endj = startj + patchSize[1];
			
					std::size_t maxIndexi =  starti;
					std::size_t maxIndexj =  startj;
					double maxVal = imageIn(starti, startj);
					for(std::size_t i0 = starti; i0 != endi; ++i0){
						for(std::size_t j0 = startj; j0 != endj; ++j0){
							if(imageIn(i0,j0) > maxVal){
								maxVal = imageIn(i0,j0);
								maxIndexi = i0;
								maxIndexj = j0;
							}
						}
					}
					//after arg-max is obtained, update gradient
					imageDer(maxIndexi, maxIndexj) += imageCoeffs(i,j);
				}
			}
		}
	}
}


}}

#endif