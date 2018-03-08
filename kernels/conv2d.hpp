/*!
 * 
 *
 * \brief       2d convolution kernel
 *
 * \author      O. Krause
 * \date        2012
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

#ifndef REMORA_KERNELS_CONV2D_HPP
#define REMORA_KERNELS_CONV2D_HPP

#include "default/conv2d.hpp"

#ifdef REMORA_USE_CLBLAST
#include "clBlast/conv2d.hpp"
#endif

namespace remora{namespace kernels{
	

///\brief Computes the convolution of a set of multi-channel images with a set of filters. 
///
/// Computes the result of applying k filters to a set of images where filters and images are allowed
/// to have multiple channels (some would call this a 3d or even 4d convolution, but we refrain from this as 
/// for two dimensions filter dimensions and image dimension must agree. E.g. it does not behave like convoluting a volume)
/// The base for the convolution is the upper left corner and there is no boundary handling, i.e. only pixels within the image area
/// are computed.
///
/// The images are stored block-row-wise. i.e. an image of size nxm with k channels is stored as 
/// and (n*k)x m matrix where n consecutive rows for the row of an image. Each image is stored as a row of the input matrix
/// Filters are stored similarly, only that in their case we have the format (n1*k*l) x m1 for a
/// set of l filters of size n1 x m1 with k channels each. the n1 rows form a channel, k*n1 rows form
/// a filter.
/// the output format is stored in the same way as image just with size (l* (m-m1+1))x(n-n1+1).
/// The caller must ensure that enough memory is available.	
template<class E1, class E2, class M, class Device>
void conv2d(
	matrix_expression<E1, Device> const& images,
	vector_expression<E2, Device> const& filter,
	matrix_expression<M, Device>& outputs,
	std::size_t num_channels,
	std::size_t num_filters,
	std::size_t image_height,
	std::size_t image_width,
	std::size_t filter_height,
	std::size_t filter_width,
	std::size_t padding_height = 0,
	std::size_t padding_width = 0
){
	std::size_t output_rows_per_filter = (image_height  - filter_height +1 + padding_height) * (image_width - filter_width +1 + padding_width);
	std::size_t filter_size = filter_width * filter_height * num_channels;
	
	REMORA_SIZE_CHECK(outputs().size1() == images().size1());
	REMORA_SIZE_CHECK(outputs().size2() == num_filters * output_rows_per_filter);
	REMORA_SIZE_CHECK(images().size2() == num_channels * image_width * image_height);
	REMORA_SIZE_CHECK(filter().size() == num_filters * filter_size);
	
	bindings::conv2d(
		images, filter, outputs, num_channels, num_filters,
		image_height, image_width, filter_height, filter_width,
		padding_height, padding_width
	);
}

}}
#endif
