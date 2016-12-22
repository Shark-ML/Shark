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

#ifndef SHARK_LINALG_BLAS_KERNELS_CONV2D_HPP
#define SHARK_LINALG_BLAS_KERNELS_CONV2D_HPP

#include "default/conv2d.hpp"


namespace shark { namespace blas {namespace kernels{
	

///\brief Computes the convolution of a multi-channel image with a set of filters. 
///
/// Computes the result of applying k filters to an image where filters and image are allowed
/// to have multiple images (some would call this a 3d or even 4d convolution, but we refrain from this as 
/// for two dimensions filter dimensions and image dimension must agree. E.g. it does not behave like convoluting a volume)
/// The base for the convolution is the upper left corner and there is no boundary handling, i.e. only pixels within the image area
/// are computed.
///
/// The image are stored block-row-wise. i.e. an image of size nxm with k channels is stored as 
/// and (n*k)x m matrix where n consecutive rows for the row of an image.
/// Filters are stored similarly, only that in their case we have the format (n1*k*l) x m1 for a
/// set of l filters of size n1 x m1 with k channels each. the n1 rows form a channel, k*n1 rows form
/// a filter.
/// the output format is stored in the same way as image just with size (l* (m-m1+1))x(n-n1+1).
/// The caller must ensure that enough memory is stored.	
template<class E1, class E2, class M>
void conv2d(
	blas::matrix_expression<E1, cpu_tag> const& image,
	blas::matrix_expression<E2, cpu_tag> const& filter,
	blas::matrix_expression<M, cpu_tag>& output,
	std::size_t num_channels,
	std::size_t num_filters
){
	SIZE_CHECK(filter().size1() % (num_filters * num_channels)  == 0);
	SIZE_CHECK(image().size1() % num_channels  == 0);
	SIZE_CHECK(output().size1() % num_filters  == 0);
	SIZE_CHECK(output().size1()/num_filters + filter().size1()/(num_filters * num_channels) -1 == image().size1() / num_channels);
	SIZE_CHECK(output().size2() + filter().size2() -1 == image().size2());
	
	bindings::conv2d(
		image, filter, output, num_channels, num_filters
	);
}

}}}
#endif
