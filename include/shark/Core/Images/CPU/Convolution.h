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

#ifndef SHARK_CORE_IMAGE_CPU_CONV_H
#define SHARK_CORE_IMAGE_CPU_CONV_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Shape.h>
#include <shark/Core/Images/Enums.h>
namespace shark{
namespace image{
	
namespace cpu{
	
template<class E1, class E2>
void im2mat(
	blas::matrix_expression<E1, blas::cpu_tag> const& images,
	blas::matrix_expression<E2, blas::cpu_tag>& mat,
	std::size_t numChannels,
	std::size_t imageHeight,
	std::size_t imageWidth,
	std::size_t numImages,
	std::size_t filterHeight,
	std::size_t filterWidth,
	ImageFormat format
){
	static_assert(std::is_same<typename E1::orientation, blas::row_major>::value, "Column major not implemented");
	static_assert(std::is_same<typename E2::orientation, blas::row_major>::value, "Column major not implemented");
	std::size_t outputHeight = imageHeight - filterHeight +1;
	std::size_t outputWidth = imageWidth - filterWidth +1;
	std::size_t rowsPerImage = outputHeight * outputWidth;
	
	if(format == ImageFormat::NCHW){
		for(std::size_t im = 0; im != numImages; ++im){
			auto image = to_matrix(row(images,im), numChannels, imageHeight * imageWidth);
			auto matImage = rows(mat, im * rowsPerImage, (im+1) * rowsPerImage);
			for(std::size_t c = 0; c != numChannels; ++c){//iterate over the channels
				auto channel = to_matrix(row(image,c), imageHeight, imageWidth);
				auto matChannel = columns(matImage, c * filterHeight * filterWidth, (c+1) * filterHeight * filterWidth );
				for(std::size_t i = 0; i != outputHeight; ++i){// iterate over row-positions in the image
					for(std::size_t j = 0; j != outputWidth; ++j){//iterate over the column-position in the image
						std::size_t rowStart = i * outputWidth + j;
						for(std::size_t i1 = 0; i1 != filterHeight; ++i1){//iterate over the the rows of the current filter
							for(std::size_t j1 = 0; j1 != filterWidth; ++j1){
								matChannel(rowStart, i1 * filterWidth + j1) = channel(i + i1, j + j1);
							}
						}
					}
				}
			}
		}
	}else if(format == ImageFormat::CNHW){
		for(std::size_t c = 0; c != numChannels; ++c){//iterate over the channels
			auto channel = to_matrix(row(images,c), numImages, imageHeight * imageWidth);
			auto matChannel = columns(mat, c * filterHeight * filterWidth, (c+1) * filterHeight * filterWidth );
			for(std::size_t im = 0; im != numImages; ++im){
				auto slice = to_matrix(row(channel,im), imageHeight, imageWidth);
				auto matSlice = rows(matChannel, im * rowsPerImage, (im+1) * rowsPerImage);
				for(std::size_t i = 0; i != outputHeight; ++i){// iterate over row-positions in the output
					for(std::size_t j = 0; j != outputWidth; ++j){//iterate over the column-position in the output
						std::size_t rowStart = i * outputWidth + j;
						for(std::size_t i1 = 0; i1 != filterHeight; ++i1){//iterate over the the rows of the current filter
							for(std::size_t j1 = 0; j1 != filterWidth; ++j1){
								matSlice(rowStart,  i1 * filterWidth + j1) = slice(i + i1, j + j1);
							}
						}
					}
				}
			}
		}
	}else{
		throw SHARKEXCEPTION("format not implemented");
	}
}

template<class E1, class E2>
void im2mat_pad(
	blas::matrix_expression<E1, blas::cpu_tag> const& images,
	blas::matrix_expression<E2, blas::cpu_tag>& mat,
	std::size_t numChannels,
	std::size_t imageHeight,
	std::size_t imageWidth,
	std::size_t numImages,
	std::size_t filterHeight,
	std::size_t filterWidth,
	ImageFormat format,
	std::size_t paddingTop,
	std::size_t paddingBottom,
	std::size_t paddingLeft,
	std::size_t paddingRight
){
	static_assert(std::is_same<typename E1::orientation, blas::row_major>::value, "Column major not implemented");
	static_assert(std::is_same<typename E2::orientation, blas::row_major>::value, "Column major not implemented");
	std::size_t imageEnd1 =  imageHeight + paddingTop;
	std::size_t imageEnd2 =  imageWidth + paddingLeft;
	std::size_t outputHeight = imageHeight - filterHeight +1 + paddingTop + paddingBottom;
	std::size_t outputWidth = imageWidth - filterWidth +1 + paddingLeft + paddingRight;
	std::size_t rowsPerImage = outputHeight * outputWidth;
	mat().clear();
	
	if(format == ImageFormat::NCHW){
		for(std::size_t im = 0; im != numImages; ++im){
			auto image = to_matrix(row(images,im), numChannels, imageHeight * imageWidth);
			auto matImage = rows(mat, im * rowsPerImage, (im+1) * rowsPerImage);
			for(std::size_t c = 0; c != numChannels; ++c){//iterate over the channels
				auto channel = to_matrix(row(image,c), imageHeight, imageWidth);
				auto matChannel = columns(matImage, c * filterHeight * filterWidth, (c+1) * filterHeight * filterWidth );
				for(std::size_t i = 0; i != outputHeight; ++i){// iterate over row-positions in the output
					for(std::size_t j = 0; j != outputWidth; ++j){//iterate over the column-position in the output
						std::size_t rowStart = i * outputWidth + j;
						for(std::size_t i1 = 0; i1 != filterHeight; ++i1){//iterate over the the rows of the current filter
							if(i1+i < paddingTop || i1+i >= imageEnd1){//top or bottom padding
								continue;
							}
							//skip the areas of the filter that are padded with zero
							std::size_t filterWidthStart = (j < paddingLeft) ? paddingLeft - j : 0;
							std::size_t filterWidthEnd = (j+filterWidth >= imageEnd2) ?  imageEnd2 - j: filterWidth;
							for(std::size_t j1 = filterWidthStart; j1 != filterWidthEnd; ++j1){
								matChannel(rowStart, i1 * filterWidth + j1) = channel(i + i1 - paddingTop, j + j1 - paddingLeft);
							}
						}
					}
				}
			}
		}
	}else if(format == ImageFormat::CNHW){
		for(std::size_t c = 0; c != numChannels; ++c){//iterate over the channels
			auto channel = to_matrix(row(images,c), numImages, imageHeight * imageWidth);
			auto matChannel = columns(mat, c * filterHeight * filterWidth, (c+1) * filterHeight * filterWidth );
			for(std::size_t im = 0; im != numImages; ++im){
				auto slice = to_matrix(row(channel,im), imageHeight, imageWidth);
				auto matSlice = rows(matChannel, im * rowsPerImage, (im+1) * rowsPerImage);
				for(std::size_t i = 0; i != outputHeight; ++i){// iterate over row-positions in the output
					for(std::size_t j = 0; j != outputWidth; ++j){//iterate over the column-position in the output
						std::size_t rowStart = i * outputWidth + j;
						for(std::size_t i1 = 0; i1 != filterHeight; ++i1){//iterate over the the rows of the current filter
							if(i1+i < paddingTop || i1+i >= imageEnd1){//top or bottom padding
								continue;
							}
							std::size_t filterWidthStart = (j < paddingLeft) ? paddingLeft - j : 0;
							std::size_t filterWidthEnd = (j+filterWidth >= imageEnd2) ?  imageEnd2 - j: filterWidth;
							for(std::size_t j1 = filterWidthStart; j1 != filterWidthEnd; ++j1){
								matSlice(rowStart,  i1 * filterWidth + j1) = slice(i + i1 - paddingTop, j + j1 - paddingLeft);
							}
						}
					}
				}
			}
		}
	}else{
		throw SHARKEXCEPTION("format not implemented");
	}
}
}

template<class M1, class M2>
void convolution(
	blas::matrix_expression<M1, blas::cpu_tag> const& images, 
	blas::matrix_expression<M1, blas::cpu_tag> const& filters, 
	blas::matrix_expression<M2, blas::cpu_tag>& results,
	Shape const& shapeImage,
	Shape const& shapeFilters,
	Shape const& shapeResults,
	std::size_t paddingHeight,
	std::size_t paddingWidth,
	ImageFormat imageFormat,
	ImageFormat filterFormat,
	bool flipFilters
){
	//~ SIZE_CHECK(shapeImage.size() == 3);
	//~ SIZE_CHECK(shapeFilters.size() == 3);
	//~ SIZE_CHECK(shapeImage[2] == shapeFilters[3] );
	//~ SIZE_CHECK(shapeImage.numElements() == images().size2());
	//~ SIZE_CHECK(shapeFilters[0] == filters().size1());
	//~ SIZE_CHECK(shapeFilters[1] * shapeFilters[2] * shapeFilters[3] == filters().size2());
	
	static_assert(std::is_same<typename M1::orientation, blas::row_major>::value, "Column major not implemented");
	static_assert(std::is_same<typename M1::storage_type::storage_tag, blas::continuous_dense_tag>::value, "Subranges not implemented");
	static_assert(std::is_same<typename M2::orientation, blas::row_major>::value, "Column major not implemented");
	typedef typename std::common_type<
		typename M1::value_type, typename M2::value_type
	>::type value_type;
	
	//extrtact geometry information
	std::size_t numImages = (imageFormat == ImageFormat::CNHW)? shapeImage[0] : images().size1();
	std::size_t numChannels = (imageFormat == ImageFormat::CNHW)? images().size1() : shapeImage[0];
	std::size_t imageHeight = shapeImage[1];
	std::size_t imageWidth = shapeImage[2];
	std::size_t numFilters = (filterFormat == ImageFormat::CNHW)? shapeFilters[0] : filters().size1();
	std::size_t filterHeight = shapeFilters[1];
	std::size_t filterWidth = shapeFilters[2];
	
	//padding. take into account that flipping the filter also flips the anchor point and thus padding
	std::size_t paddingTop = flipFilters? paddingHeight - paddingHeight / 2: paddingHeight / 2;
	std::size_t paddingBottom = paddingHeight - paddingTop;
	std::size_t paddingLeft = flipFilters? paddingWidth - paddingWidth / 2: paddingWidth / 2;
	std::size_t paddingRight = paddingWidth - paddingLeft;
	
	//sizes
	std::size_t outputRowsPerFilter = (imageHeight  - filterHeight +1 + paddingHeight) * (imageWidth - filterWidth +1 + paddingWidth);
	std::size_t filterSize = filterWidth * filterHeight * numChannels;
	
	//allocate storage and create temporary matrices
	boost::alignment::aligned_allocator<value_type,64> allocator;
	value_type* imageStorage = allocator.allocate( numImages * outputRowsPerFilter * filterSize);
	value_type* filterStorage = allocator.allocate(numFilters * filterSize);
	blas::dense_matrix_adaptor<value_type, blas::row_major, blas::continuous_dense_tag> mat(imageStorage,numImages * outputRowsPerFilter, filterSize);
	blas::dense_matrix_adaptor<value_type> filter_transformed(filterStorage, numFilters, filterSize);
	
	//copy filters to temporary storage
	//apply flipping and converison to NCHW storage if encessary
	if(!flipFilters && filterFormat == ImageFormat::NCHW){//natural format
		for(std::size_t f = 0; f != numFilters; ++f){
			for(std::size_t i = 0; i != filterSize; ++i){
				filter_transformed(f,i) = filters()(f, i);
			}
			
		}
	}else if(flipFilters && filterFormat == ImageFormat::NCHW){//only flip the width and height coordinates
		for(std::size_t f = 0; f != numFilters; ++f){
			std::size_t e =  0;
			for(std::size_t c = 0; c != numChannels; ++c){
				for(std::size_t i = 0; i != filterHeight; ++i){
					for(std::size_t j = 0; j != filterWidth; ++j, ++e){
						std::size_t pixelIdx = (filterHeight - i - 1) * filterWidth + (filterWidth - j - 1);
						filter_transformed(f,e) = filters()(f, c * filterWidth * filterHeight + pixelIdx);
					}
				}
			}
		}
	}else if(!flipFilters && filterFormat == ImageFormat::CNHW){//only exchange N and C index
		std::size_t numPixels = filterWidth * filterHeight;
		for(std::size_t f = 0; f != numFilters; ++f){
			std::size_t e =  0;
			for(std::size_t c = 0; c != numChannels; ++c){
				for(std::size_t i = 0; i != numPixels; ++i, ++e){
					filter_transformed(f,e) = filters()(c, f * numPixels + i);
				}
			}
		}
		//~ std::cout<<filters<<std::endl;
		//~ std::cout<<filter_transformed<<std::endl;
		//~ std::terminate();
	}else if(flipFilters && filterFormat == ImageFormat::CNHW){//perform both transformations
		std::size_t numPixels = filterWidth * filterHeight;
		for(std::size_t f = 0; f != numFilters; ++f){
			std::size_t e =  0;
			for(std::size_t c = 0; c != numChannels; ++c){
				for(std::size_t i = 0; i != filterHeight; ++i){
					for(std::size_t j = 0; j != filterWidth; ++j, ++e){
						std::size_t pixelIdx = (filterHeight - i - 1) * filterWidth + (filterWidth - j - 1);
						filter_transformed(f,e) = filters()(c, f * numPixels + pixelIdx);
					}
				}
			}
		}
	}
	
	//create matrix representation for convolution. This will always create a matrix 
	// with rows in NHW format and columns in CHW
	if(paddingHeight == 0 || paddingWidth == 0){
		cpu::im2mat(
			images, mat, numChannels, 
			imageHeight, imageWidth, numImages, filterHeight, filterWidth, 
			imageFormat
		);
	}else{
		cpu::im2mat_pad(
			images, mat, numChannels, 
			imageHeight, imageWidth, numImages, filterHeight, filterWidth, imageFormat,
			paddingTop, paddingBottom, paddingLeft, paddingRight
		);
	}
	
	if(imageFormat == ImageFormat::CNHW){
		blas::dense_matrix_adaptor<value_type> output_transformed(results().raw_storage().values, numFilters, numImages * outputRowsPerFilter);
		noalias(output_transformed) += filter_transformed % trans(mat);
	}else if(imageFormat == ImageFormat::NCHW){
		blas::dense_matrix_adaptor<value_type, blas::row_major, blas::continuous_dense_tag> output_transformed(
			results().raw_storage().values, numImages * numFilters, outputRowsPerFilter
		);
		for(std::size_t im = 0; im != numImages; ++im){
			auto imageMat = rows(mat, im * outputRowsPerFilter, (im+1) * outputRowsPerFilter);
			auto output = rows(output_transformed, im * numFilters, (im+1) * numFilters);
			
			noalias(output) += filter_transformed % trans(imageMat);
		}
	}
	//deallocate storage
	allocator.deallocate(imageStorage,numImages * outputRowsPerFilter * filterSize);
	allocator.deallocate(filterStorage, numFilters * filterSize);
}


template<class M1, class M2>
void convolutionBackwardInputs(
	blas::matrix_expression<M1, blas::cpu_tag> const& delta, 
	blas::matrix_expression<M1, blas::cpu_tag> const& filters, 
	blas::matrix_expression<M2, blas::cpu_tag>& derivatives,
	Shape const& shapeDelta,
	Shape const& shapeFilters,
	Shape const& shapeDerivatives,
	std::size_t paddingHeight,
	std::size_t paddingWidth,
	bool flipFilters
){
	paddingHeight = 2*(shapeFilters[1] - 1) - paddingHeight;
	paddingWidth = 2*(shapeFilters[2] - 1) - paddingWidth;
	convolution(
		delta, filters, derivatives,
		shapeDelta, shapeFilters, shapeDerivatives,
		paddingHeight, paddingWidth,
		ImageFormat::NCHW, ImageFormat::CNHW, !flipFilters
	);
}

template<class M1, class M2>
void convolutionBackwardFilters(
	blas::matrix_expression<M1, blas::cpu_tag> const& images,
	blas::matrix_expression<M1, blas::cpu_tag> const& delta,
	blas::matrix_expression<M2, blas::cpu_tag>& derivatives,
	Shape const& shapeImage,
	Shape const& shapeDelta,
	Shape const& shapeDerivatives,
	std::size_t paddingHeight,
	std::size_t paddingWidth,
	bool flipFilters
){
	image::convolution(
		images, delta, derivatives,
		shapeImage, shapeDelta, shapeDerivatives,
		paddingHeight, paddingWidth,
		ImageFormat::CNHW, ImageFormat::CNHW, flipFilters
	);
}

}}

#endif