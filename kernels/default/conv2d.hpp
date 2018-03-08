/*!
 *
 *
 * \brief       Implements the 2D convolution kernel for cpus
 *
 * \author      O. Krause
 * \date        2016
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
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

#ifndef REMORA_KERNELS_DEFAULT_Conv2D_HPP
#define REMORA_KERNELS_DEFAULT_Conv2D_HPP

#include "simd.hpp"
#include "../gemm.hpp"
#include <type_traits> //for std::common_type and aligned storage
namespace remora{namespace bindings {
	

/// \brief Transforms the given image into a row-major format for convolution 
///
/// The resulting matrix has one row for each output of the convolution.
/// each row contains the patch used for computing the result.	
template<class E1, class E2>
void im2mat(
	matrix_expression<E1, cpu_tag> const& images,
	matrix_expression<E2, cpu_tag>& output,
	std::size_t num_channels,
	std::size_t image_height,
	std::size_t image_width,
	std::size_t filter_height,
	std::size_t filter_width
){
	static_assert(std::is_same<typename E1::orientation, row_major>::value, "Column major not implemented");
	static_assert(std::is_same<typename E2::orientation, row_major>::value, "Column major not implemented");
	std::size_t rows_per_image = (image_height - filter_height +1) * (image_width - filter_width +1);
	//the order of loops is chosen such, that only very little changes of rows are performed
	for(std::size_t im = 0; im != images().size1(); ++im){
		for(std::size_t i = 0; i != image_height - filter_height +1; ++i){// iterate over row-positions in the image
			for(std::size_t i1 = 0; i1 != filter_height; ++i1){//iterate over the the rows of the current filter
				for(std::size_t j = 0; j != image_width - filter_width +1; ++j){//iterate over the column-position in the image
					std::size_t row_start = im * rows_per_image + i * (image_width - filter_width +1) + j;
					for(std::size_t j1 = 0; j1 != filter_width; ++j1){
						std::size_t col_start = (i1 * filter_width + j1) * num_channels;
						std::size_t image_start = ((i+i1) * image_width + j+j1) * num_channels;
						for(std::size_t c = 0; c != num_channels; ++c){//iterate over the channels
							output()(row_start, col_start + c) = images()(im,image_start + c);
						}
					}
				}
			}
		}
	}
}

template<class E1, class E2>
void im2mat_pad(
	matrix_expression<E1, cpu_tag> const& images,
	matrix_expression<E2, cpu_tag>& output,
	std::size_t num_channels,
	std::size_t image_height,
	std::size_t image_width,
	std::size_t filter_height,
	std::size_t filter_width,
	std::size_t padding_height,
	std::size_t padding_width
){
	static_assert(std::is_same<typename E1::orientation, row_major>::value, "Column major not implemented");
	static_assert(std::is_same<typename E2::orientation, row_major>::value, "Column major not implemented");
	std::size_t image_start1 = padding_height/2;
	std::size_t image_end1 =  image_height + image_start1;
	std::size_t image_start2 = padding_width/2;
	std::size_t image_end2 =  image_width + image_start2;
	std::size_t output_width = image_width - filter_width + 1 + padding_width;
	std::size_t output_height = image_height - filter_height + 1 + padding_height;
	std::size_t rows_per_image =output_width * output_height;
	//the order of loops is chosen such, that only very little changes of rows are performed
	for(std::size_t im = 0; im != images().size1(); ++im){
		for(std::size_t i = 0; i != output_height; ++i){// iterate over row-positions in the output
			for(std::size_t i1 =  0; i1 != filter_height; ++i1){//iterate over the the rows of the current filter
				if(i1+i < image_start1 || i1+i >= image_end1){//special case: we are on the border above or below
					for(std::size_t j = 0; j != output_width; ++j){//iterate over the column-position in the output
						std::size_t row_start = im * rows_per_image + i * output_width +j;
						std::size_t col_start = i1 * filter_width * num_channels;
						for(std::size_t c = 0; c != num_channels * filter_width; ++c){//iterate over the channels
							output()(row_start, col_start + c) = 0;
						}
					}
					continue;//no need to got on, we are done
				}
				for(std::size_t j = 0; j != output_width; ++j){//iterate over the column-position in the output
					std::size_t row_start = im * rows_per_image + i * output_width + j;
					for(std::size_t j1 = 0; j1 != filter_width; ++j1){
						std::size_t col_start = (i1 * filter_width + j1) * num_channels;
						
						if(j+j1 < image_start2 || j+j1 >= image_end2){
							for(std::size_t c = 0; c != num_channels; ++c){//iterate over the channels
								output()(row_start, col_start + c) = 0;
							}
						}else{
							std::size_t image_start = ((i+i1-image_start1) * image_width + j+j1-image_start2) * num_channels;
							for(std::size_t c = 0; c != num_channels; ++c){//iterate over the channels
								output()(row_start, col_start + c) = images()(im,image_start + c);
							}
						}
					}
				}
			}
		}
	}
}


template<class E1, class E2, class M>
void conv2d(
	matrix_expression<E1, cpu_tag> const& images,
	vector_expression<E2, cpu_tag> const& filter,
	matrix_expression<M, cpu_tag>& outputs,
	std::size_t num_channels,
	std::size_t num_filters,
	std::size_t image_height,
	std::size_t image_width,
	std::size_t filter_height,
	std::size_t filter_width,
	std::size_t padding_height,
	std::size_t padding_width
){
	static_assert(std::is_same<typename E1::orientation, row_major>::value, "Column major not implemented");
	static_assert(std::is_same<typename E1::storage_type::storage_tag, continuous_dense_tag>::value, "Subranges not implemented");
	static_assert(std::is_same<typename M::orientation, row_major>::value, "Column major not implemented");
	typedef typename std::common_type<
		typename E1::value_type, typename E2::value_type, typename M::value_type
	>::type value_type;
	
	std::size_t output_rows_per_filter = (image_height  - filter_height +1 + padding_height) * (image_width - filter_width +1 + padding_width);
	std::size_t filter_size = filter_width * filter_height * num_channels;
	std::size_t num_images = images().size1();
	
	REMORA_SIZE_CHECK(outputs().size1() == images().size1());
	REMORA_SIZE_CHECK(outputs().size2() == num_filters * output_rows_per_filter);
	REMORA_SIZE_CHECK(images().size2() == num_channels * image_width * image_height);
	REMORA_SIZE_CHECK(filter().size() == num_filters * filter_size);
	
	//allocate storage and create temporary matrices
	boost::alignment::aligned_allocator<value_type,64> allocator;
	value_type* image_storage = allocator.allocate( num_images * output_rows_per_filter * filter_size);
	value_type* filter_storage = allocator.allocate(num_filters * filter_size);
	dense_matrix_adaptor<value_type, row_major, cpu_tag> image_transformed(image_storage,num_images * output_rows_per_filter, filter_size);
	dense_matrix_adaptor<value_type, row_major, cpu_tag> filter_transformed(filter_storage, num_filters, filter_size);
	dense_matrix_adaptor<value_type, row_major, cpu_tag> output_transformed(outputs().raw_storage().values, num_images * output_rows_per_filter, num_filters);
	//copy image to temporary storage
	if(padding_height == 0 && padding_width == 0){
		im2mat(images,image_transformed, num_channels, image_height, image_width, filter_height, filter_width);
	}else{
		im2mat_pad(images,image_transformed, num_channels, image_height, image_width, filter_height, filter_width, padding_height, padding_width);
	}
	//copy filters to temporary storage
	for(std::size_t f = 0; f != num_filters; ++f){
		for(std::size_t i = 0; i != filter_size; ++i){
			filter_transformed(f,i) = filter()(f * filter_size + i);
		}
	}
	
	//do the computation
	kernels::gemm(image_transformed, trans(filter_transformed), output_transformed, value_type(1.0));
	
	//deallocate storage
	allocator.deallocate(image_storage,num_images * output_rows_per_filter * filter_size);
	allocator.deallocate(filter_storage, num_filters * filter_size);
}

}}

#endif
