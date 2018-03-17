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
	std::size_t depth = shape[2];
	std::size_t outputHeight = shape[0]/patchSize[0];
	std::size_t outputWidth = shape[1]/patchSize[1];
	std::size_t outputPixels = outputWidth * outputHeight;
	SIZE_CHECK(outputs.size2() == outputPixels * depth);
	
	//for all images
	for(std::size_t img = 0; img != inputs.size1(); ++img){
		//Extract single images and create matrices (pixels,channels)
		auto imageIn = to_matrix(row(inputs,img), shape[0] * shape[1], depth);
		auto imageOut = to_matrix(row(outputs,img), outputHeight * outputWidth, depth);
		//traverse over all pixels of the output image
		for(std::size_t p = 0; p != outputPixels; ++p){
			auto pixel = row(imageOut,p);
			//extract pixel coordinates in input image
			std::size_t starti = (p / outputWidth) * patchSize[0];
			std::size_t startj = (p % outputWidth) * patchSize[1];
			std::size_t endi = starti + patchSize[0];
			std::size_t endj = startj + patchSize[1];
			
			//traverse the patch on the input image and compute maximum
			noalias(pixel) = row(imageIn, starti * shape[1] + startj);
			for(std::size_t i = starti; i != endi; ++i){
				for(std::size_t j = startj; j != endj; ++j){
					std::size_t index = i * shape[1] + j;
					noalias(pixel) = max(pixel,row(imageIn, index));
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
	std::size_t depth = shape[2];
	std::size_t outputHeight = shape[0]/patchSize[0];
	std::size_t outputWidth = shape[1]/patchSize[1];
	std::size_t outputPixels = outputWidth * outputHeight;
	SIZE_CHECK(coefficients.size2() == outputPixels * depth);
	//for all images
	for(std::size_t img = 0; img != inputs.size1(); ++img){
		//Extract single images and create matrices (pixels,channels)
		auto imageCoeffs = to_matrix(row(coefficients,img),  outputHeight * outputWidth, depth);
		auto imageIn = to_matrix(row(inputs,img), shape[0] * shape[1], depth);
		auto imageDer = to_matrix(row(derivatives,img), shape[0] * shape[1], depth);
		//traverse over all pixels of the output image
		for(std::size_t p = 0; p != outputPixels; ++p){
			std::size_t starti = (p / outputWidth) * patchSize[0];
			std::size_t startj = (p % outputWidth) * patchSize[1];
			std::size_t endi = starti + patchSize[0];
			std::size_t endj = startj + patchSize[1];
			
			//traverse the patch on the input image and compute arg-max for each channel
			for(std::size_t c = 0; c != depth; ++c){
				std::size_t maxIndex =  starti * shape[1] + startj;
				double maxVal = imageIn(maxIndex,c);
				for(std::size_t i = starti; i != endi; ++i){
					for(std::size_t j = startj; j != endj; ++j){
						std::size_t index = i * shape[1] + j;
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


}}

#endif