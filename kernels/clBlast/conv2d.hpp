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

#ifndef REMORA_KERNELS_CLBLAST_CONV2D_HPP
#define REMORA_KERNELS_CLBLAST_CONV2D_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <clblast.h>
namespace remora{namespace bindings {
		
template<class E1, class E2, class M>
void conv2d(
	matrix_expression<E1, gpu_tag> const& images,
	vector_expression<E2, gpu_tag> const& filter,
	matrix_expression<M, gpu_tag>& outputs,
	std::size_t num_channels,
	std::size_t num_filters,
	std::size_t image_height,
	std::size_t image_width,
	std::size_t filter_height,
	std::size_t filter_width,
	std::size_t padding_height,
	std::size_t padding_width
){
	static_assert(std::is_same<typename E1::orientation, row_major>::value, "[conv2d] Column major not implemented");
	static_assert(std::is_same<typename E1::storage_type::storage_tag, continuous_dense_tag>::value, "[conv2d] Subranges not implemented");
	static_assert(std::is_same<typename M::orientation, row_major>::value, "[conv2d] Column major not implemented");
	
	static_assert(std::is_same<typename E1::value_type, typename E2::value_type>::value, "[conv2d] Arguments do not have same value type");
	static_assert(std::is_same<typename E1::value_type, typename M::value_type>::value, "[conv2d] Arguments do not have same value type");
	
	static_assert(std::is_base_of<dense_tag, typename E1::evaluation_category::tag>::value, "[conv2d] images is not dense");
	static_assert(std::is_base_of<continuous_dense_tag, typename E2::storage_type::storage_tag>::value, "[conv2d] filter does not have continuous dense storage layout");
	static_assert(std::is_base_of<continuous_dense_tag, typename M::storage_type::storage_tag>::value, "[conv2d] outputs does not have dense storage layout");
	

	typedef typename E1::value_type value_type;
	
	//pre-evaluate images into a temporary if necessary
	auto const& images_eval = eval_expression(images);	
	
	//~ if(padding_height != 0 || padding_width != 0){
		//~ //note: the below is simply creating a temporary matrix and copying the image into the subranges. 
		//~ //However we can not use matrix proxy expressions directly and instead need to rely on this slightly 
		//~ //nasty loop
		//~ auto context = outputs().queue().get_context();
		//~ std::size_t padded_width = image_width+padding_width;
		//~ std::size_t paddedSize = padded_width * (image_height + padding_height) * num_channels;
		//~ //create storage for the image
		//~ gpu::dense_matrix_storage<value_type, dense_tag> storage = {{context, images().size1() * paddedSize},0,paddedSize};
		//~ dense_matrix_adaptor<value_type, row_major, gpu_tag> padded_images(storage,images().size1(), paddedSize);
		//~ adaptor.clear();//initialize padding to 0.
		//~ for(std::size_t im = 0; im != images().size1(); ++im){
			//~ //cut out the i-th image starting at the first non-zero element
			//~ //i.e. we skip the initial 0-rows and the first few nonzero element of the first real rows
			//~ std::size_t offset = im * paddedSize + ((padding_height/2) * padded_width  + padding_width/2) * num_channels;
			//~ gpu::dense_matrix_storage<value_type, dense_tag> sub_storage = {storage.context, offset, padded_width * num_channels};
			//~ dense_matrix_adaptor<value_type, row_major, gpu_tag> sub_image(sub_storage,image_height, image_width * num_channels);
			//~ noalias(sub_image) = row(images,im);
		//~ }
		
		//~ conv2d(
			//~ padded_images,filter,outputs,
			//~ num_channels, num_filters, 
		//~ )
	//~ }
	
	
	
	std::size_t output_height = (image_height  - filter_height +1 + padding_height);
	std::size_t output_width = (image_width - filter_width +1 + padding_width);
	std::size_t filter_size = filter_width * filter_height * num_channels;
	std::size_t num_images = images().size1();
	
	REMORA_SIZE_CHECK(outputs().size1() == images().size1());
	REMORA_SIZE_CHECK(outputs().size2() == num_filters * output_width * output_height);
	REMORA_SIZE_CHECK(images().size2() == num_channels * image_width * image_height);
	REMORA_SIZE_CHECK(filter().size() == num_filters * filter_size);
	
	// for this implementation, we use the GemmBatched extension to BLAS implemented by clBlast
	// GemmBatched allows to perform a set of matrix-matrix multiplications
	// C_i= alpha_i * C_i + beta_i * A_i * B_i
	// in parallel.
	// to use this, note that our filter in memory layout is stored as
	// num_filters x filter_height x filter_width x num_channels
	// where num_channels is the fastest index (i.e. continuous in memory)
	// and num_filters the slowest. The dense layout allows us to 
	// get one matrix-sized slice of this. 
	// We know that the (filter_width x num_channels)-matrix is continuous
	// in memory in both filter and image. So we can linearize this part
	// and treat it as (filter_width * num_channels)-dim vector.
	// Then we can fix the filter_height variable to obtain a 3-tensor
	// which is linearized a matrix of size num_filters x (filter_width * num_channels).
	// like this, we obtain one 3-tensor for each value of filter_height, this is usually not too large.
	// this allows in the image to extract fitting slices of size (filter_width * num_channels)
	// simply by moving the vector element by element over the matrix.
	// we can again parallelize over the images to obtain sub-matrices of size
	// num_images x (filter_width * num_channels).
	// like this, for each filter slice, we obtain output_width * output_height
	// matrix-matrix multiplications which are then repeated over each slice of the filter.
	//We compute slices one-after another and add them up on the outputs matrix.
	
	//obtain matrix storage
	auto storage_images = images_eval.raw_storage();
	auto storage_filter = filter().raw_storage();
	auto storage_outputs = outputs().raw_storage();
	
	//setup required constants for offsets
	std::size_t num_multiplications = output_width * output_height;
	std::vector<value_type> alphas(num_multiplications,value_type(1));
	std::vector<value_type> const& betas = alphas;
	//outputs offsets are constant
	std::vector<std::size_t> outputs_offsets(num_multiplications);
	std::vector<std::size_t> im_offsets(num_multiplications,0);
	std::vector<std::size_t> filter_offsets(num_multiplications,0);
	for(std::size_t i = 0; i != output_height; ++i){
		for(std::size_t j = 0; j != output_width; ++j){
			std::size_t index = i * output_width + j;
			outputs_offsets[index] = index * num_filters;
		}
	}
	
	for(std::size_t k = 0; k != filter_height; ++k){
		
		for(std::size_t i = 0; i != output_height; ++i){
			for(std::size_t j = 0; j != output_width; ++j){
				std::size_t index = i * output_width + j;
				im_offsets[index] = ((i+k) * image_width + j) * num_channels;
			}
		}
		//call GemmBatches
		using namespace clblast;
		cl_event* event = nullptr;//todo: store events for out-of-order queues 
		auto status = GemmBatched<value_type>(
			Layout::kRowMajor, Transpose::kNo, Transpose::kYes,
			num_images, num_filters, num_channels * filter_width,
			alphas.data(),
			storage_images.buffer.get(), im_offsets.data(), storage_images.leading_dimension,
			storage_filter.buffer.get(), filter_offsets.data(), filter_size,
			betas.data(),
			storage_outputs.buffer.get(), outputs_offsets.data(), storage_outputs.leading_dimension,
			num_multiplications,
			&outputs().queue().get(), event
		);
		//~ std::cout<<(int)status<<std::endl;
		assert(status == StatusCode::kSuccess);
		if(k < filter_height -1){
			for(auto& offset: filter_offsets){
				offset += num_channels * filter_width;
			}
		}
	}
	
	
}

}}

#endif
