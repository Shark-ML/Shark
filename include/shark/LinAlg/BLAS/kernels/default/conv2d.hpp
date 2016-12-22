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

#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_Conv2D_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_Conv2D_HPP

#include "../../detail/matrix_proxy_classes.hpp"
#include <boost/align/aligned_allocator.hpp> //mgemm requires aligned allocations
#include <type_traits> //for std::common_type and aligned storage
namespace shark {namespace blas {namespace bindings {

template <typename T>
struct conv2d_block_size {
	static const unsigned vector_length = SHARK_BLAS_VECTOR_LENGTH/sizeof(T); // Number of elements in a vector register
	static const unsigned col_block_size = 3; //number of neighbouring columbs to be computed in a mini block
	static const unsigned num_filter_outputs = 2*vector_length; //number of filters to be computed in a block. must be multiple of vector_length
	static const unsigned output_block_size = (200/col_block_size) * col_block_size; // maximum size of the tile of an output. Bigger is not always better.
	static const unsigned align = 64; //alignment of the memory
};


// convolution micro kernel
// computes the convolution of single channel image with a set of filters
// the following is assumed: 
// image is an image with a single channel of size image_size1 * image_size2 stored in row-major format
// filter is an image of size filter_size1x filter_size2 where every pixel consists of block_size::num_filter_outputs
// values standing for the outputs of the filter. those valeus are stored interleaved.
// output has enough size to store the result of the convolution in the same interleaved format as the
// the filter
// note that it must hold (image_size2 - filter_size2 +1)/ block_size::col_block_size == 0
// and all arrays must be aligned to block_size::align boundary.
//
// see conv2d_block_size for the tuning parameters.
template<class T, class block_size>
void uConv2d(T const* image, T const* filter, T* output,
	std::size_t image_size1, std::size_t image_size2,
	std::size_t filter_size1, std::size_t filter_size2,
	block_size
){
	BOOST_ALIGN_ASSUME_ALIGNED(image, block_size::align);
	BOOST_ALIGN_ASSUME_ALIGNED(filter, block_size::align);
	BOOST_ALIGN_ASSUME_ALIGNED(output, block_size::align);
	static std::size_t const col_block_size = block_size::col_block_size;
	static std::size_t const num_filter_outputs = block_size::num_filter_outputs;
	
#ifdef SHARK_USE_SIMD
	static const std::size_t num_filter_vec = num_filter_outputs/block_size::vector_length;
#ifdef BOOST_COMP_CLANG_DETECTION
	typedef T vx __attribute__((ext_vector_type (block_size::vector_length)));
#else
	typedef T vx __attribute__((vector_size (block_size::vector_length * sizeof(T))));
#endif
#else
	typedef T vx;
	static const std::size_t num_filter_vec = num_filter_outputs;
#endif
	vx* output_packed = (vx*) output; 
	std::size_t output_size1 = image_size1 - filter_size1 + 1;
	std::size_t output_size2 = image_size2 - filter_size2 + 1;
	
	for(std::size_t i = 0; i != output_size1; ++i){
		for(std::size_t j = 0; j != output_size2; j += col_block_size){//we hand unroll this loop, thus output_size2 must be divisable by col_block_size
			//create local accumulator register
#ifdef SHARK_USE_SIMD
			vx acc[num_filter_vec * col_block_size] = {};
#else
			typename std::aligned_storage<sizeof(T[num_filter_outputs * col_block_size]),block_size::align>::type Pa;
			T* acc = reinterpret_cast<T*>(&Pa);
			for (std::size_t l = 0; l < num_filter_vec * col_block_size; l++)
				acc[l] = 0;
#endif
			//perform the convolution with result (i,j)
			vx const* filter_packed = (vx const*) filter;
			for(std::size_t i0 = 0; i0 != filter_size1; ++i0){
				T const* imageP = image + (i+i0) * image_size2 + j;
				for(std::size_t j0 = 0; j0 != filter_size2; ++j0){
					for(std::size_t j1 = 0; j1 != col_block_size; ++j1){
						T val = imageP[j0+j1];
						for(std::size_t k0 = 0; k0 < num_filter_vec; ++k0){
							acc[ j1 * num_filter_vec + k0] += val * filter_packed[k0];
						}
					}
					filter_packed += num_filter_vec;
				}
			}
			//add result to packed storage
			for(std::size_t j1 = 0; j1 != col_block_size; ++j1){
				for (std::size_t k0 = 0; k0 < num_filter_vec; ++k0){
					*output_packed += acc[ j1 * num_filter_vec + k0];
					++output_packed;
					
				}
			}
		}
	}
}

//takes a filter and transforms it into block-interleaved format as described above
template<class T, class E, class block_size>
void pack_filter(
	T* p, blas::matrix_expression<E, blas::cpu_tag> const& filter_im, std::size_t num_channels, std::size_t num_filters, 
	block_size
){
	static std::size_t const num_filter_outputs = block_size::num_filter_outputs;
	std::size_t size1 = filter_im().size1()/num_channels/num_filters;
	std::size_t size2 = filter_im().size2();
	
	std::size_t filter_blocks = (num_filters + num_filter_outputs - 1)/num_filter_outputs;
	std::size_t filter_packed_stride2 = num_filter_outputs;
	std::size_t filter_packed_stride1 = size2 * filter_packed_stride2;
	
	for( std::size_t filter_block = 0; filter_block != filter_blocks; ++filter_block){
		for(std::size_t channel = 0; channel != num_channels; ++channel){
			for(std::size_t f = 0; f != num_filter_outputs; ++f){
				std::size_t filter = filter_block * num_filter_outputs + f;
				//obtain proxy to the target memory which is used to store this channel
				blas::dense_matrix_adaptor<T> filter_packed_channel(p + f,size1,size2, filter_packed_stride1, filter_packed_stride2);
				if(filter >= num_filters){//check if the filter is padding and pad with 0
					filter_packed_channel.clear();
				}else{
					//obtain proxy to current image channel and assign it(handles case of E being column major)
					std::size_t filter_channel_start = (filter * num_channels + channel) * size1;
					blas::matrix_range<typename blas::const_expression<E>::type > filter_channel(filter_im(),filter_channel_start,filter_channel_start+size1, 0,size2);
					noalias(filter_packed_channel) = filter_channel;
				}
			}
			p += num_filter_outputs * size1 * size2;//move memory to next channel block
		}
	}
}

//takes a tile of the image and copies it into a temporary array. if the tile exceeds the image boundaries, those values are undefined.
template<class T, class E, class block_size>
void pack_image(
	T* p, blas::matrix_expression<E, blas::cpu_tag> const& image, std::size_t num_channels,
	std::size_t start1, std::size_t start2,
	std::size_t size1, std::size_t size2,
	block_size
){	
	//the padded region does not matter as pixels affected by it are thrown away
	std::size_t unpadded_size2 = std::min(size2, image().size2() - start2);
	
	for(std::size_t channel = 0; channel != num_channels; ++channel){
		std::size_t image_channel_start1 = channel * image().size1()/num_channels +start1;

		//obtain proxy to the unpadded target memory which is used to store this channel
		blas::dense_matrix_adaptor<T> image_packed_channel(p, size1, unpadded_size2, size2, 1);

		//obtain proxy to current image channel and assign it(handles case of E being column major)
		blas::matrix_range<typename blas::const_expression<E>::type > image_channel(image(),image_channel_start1,image_channel_start1 + size1, start2, start2 + unpadded_size2);
		noalias(image_packed_channel) = image_channel;
		
		//move pointer to next channel
		p += size1 * size2;
	}
}

template<class E1, class E2, class M>
void conv2d(
	blas::matrix_expression<E1, blas::cpu_tag> const& image,
	blas::matrix_expression<E2, blas::cpu_tag> const& filter,
	blas::matrix_expression<M, blas::cpu_tag>& output,
	std::size_t num_channels,
	std::size_t num_filters
){	
	typedef typename std::common_type<typename E1::value_type, typename E2::value_type>::type value_type;
	typedef conv2d_block_size<value_type> block_size;
		
	static std::size_t const col_block_size = block_size::col_block_size;
	static std::size_t const num_filter_outputs = block_size::num_filter_outputs;
	static std::size_t const output_block_size = block_size::output_block_size;
	
	std::size_t filter_blocks = (num_filters + num_filter_outputs-1)/num_filter_outputs;
	std::size_t filter_size1 = filter().size1()/(num_channels * num_filters);
	std::size_t filter_size2 = filter().size2();
	std::size_t image_size1 = image().size1()/num_channels;
	std::size_t image_size2 = image().size2();
	std::size_t output_size1 = image_size1 +1 - filter_size1;
	std::size_t output_size2 = image_size2 +1 - filter_size2;
	std::size_t image_block_size = output_block_size + std::max(filter_size1,filter_size2) - 1;
	std::size_t size_filter_block = num_filter_outputs * filter_size1 * filter_size2;
	std::size_t output_blocks1 = (output_size1+output_block_size - 1)/output_block_size;
	std::size_t output_blocks2 = (output_size2+output_block_size - 1)/output_block_size;
	
	boost::alignment::aligned_allocator<value_type,block_size::align> allocator;
	value_type* filter_temporary = allocator.allocate(filter_blocks * num_channels * size_filter_block);
	value_type* image_temporary = allocator.allocate(num_channels * image_block_size * image_block_size);
	value_type* output_temporary = allocator.allocate(num_filter_outputs * output_block_size * output_block_size);
	
	//pack filter into temporary memory
	pack_filter(filter_temporary,filter,num_channels, num_filters, block_size());
	
	for(std::size_t i = 0; i != output_blocks1; ++i){
		std::size_t cur_out_size1 = std::min(output_block_size, output_size1 - i * output_block_size);
		std::size_t cur_image_size1 = cur_out_size1 + filter_size1 -1;//extend image area to fit the patch
		std::size_t block_start1 = i * output_block_size;
		for(std::size_t j = 0; j != output_blocks2; ++j){
			std::size_t cur_out_size2 = std::min(output_block_size,output_size2 - j * output_block_size);
			cur_out_size2 = (cur_out_size2 +col_block_size -1)/col_block_size * col_block_size;//take padding into account
			std::size_t cur_image_size2 = cur_out_size2 + filter_size2 -1;//extend image area to fit the patch
			std::size_t block_start2 = j * output_block_size;
	
			pack_image(
				image_temporary, image, num_channels,
				block_start1,block_start2,
				cur_image_size1, cur_image_size2,
				block_size()
			);
			
			for( std::size_t block = 0; block != filter_blocks; ++block){
				//clear temporary output memory
				for(std::size_t l = 0; l != num_filter_outputs * cur_out_size1 * cur_out_size2; ++l){
					output_temporary[l] = value_type();
				}
				for(std::size_t channel = 0; channel != num_channels; ++channel){
					value_type const* image_block = image_temporary + channel * cur_image_size1 * cur_image_size2;
					value_type const* filter_block = filter_temporary + (block * num_channels + channel) * size_filter_block;
					uConv2d(image_block, filter_block, output_temporary,
						cur_image_size1, cur_image_size2, filter_size1, filter_size2, block_size()
					);
				}
				
				//copy result to output
				std::size_t unpadded_size2 = std::min(cur_out_size2,output_size2 - block_start2);
				for(std::size_t f = 0; f != num_filter_outputs; ++f){
					std::size_t filter_index = block * num_filter_outputs + f;
					if(filter_index >= num_filters) break; // do not copy padding
					std::size_t output_start1 = filter_index * output_size1 + block_start1;
					//obtain proxy to the unpadded target memory which is used to store this channel
					blas::dense_matrix_adaptor<value_type const> output_packed_channel(output_temporary + f, cur_out_size1, unpadded_size2, cur_out_size2 * num_filter_outputs, num_filter_outputs);

					//obtain proxy to current image channel and assign it(handles case of E being column major)
					blas::matrix_range<M> output_channel(output(),output_start1,output_start1 + cur_out_size1, block_start2, block_start2 + unpadded_size2);
					noalias(output_channel) = output_packed_channel;
				}
			}
		}
	}
	allocator.deallocate(filter_temporary, filter_blocks * num_channels * size_filter_block);
	allocator.deallocate(image_temporary, num_channels * image_block_size * image_block_size);
	allocator.deallocate(output_temporary, num_filter_outputs * output_size1 * output_size2);
}

}}}

#endif
