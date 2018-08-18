//===========================================================================
/*!
 * 
 *
 * \brief       Routines to Reorder the Shape of Images
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


#ifndef SHARK_CORE_IMAGE_REORDER_H
#define SHARK_CORE_IMAGE_REORDER_H

#include <shark/LinAlg/Base.h>
#include "CPU/Reorder.h"
#ifdef SHARK_USE_OPENCL
#include "OpenCL/Reorder.h"
#endif
namespace shark{
enum class ImageFormat{
	NHWC = 1234,
	NWHC = 1324,
	NCHW = 1423,
	NCWH = 1432,
	CHWN = 4231
};


namespace image{
template<class T, class Device>
void reorder(
	blas::dense_vector_adaptor<T const, blas::continuous_dense_tag, Device> input, 
	blas::dense_vector_adaptor<T, blas::continuous_dense_tag, Device> output, 
	Shape const& shapeIn,
	ImageFormat orderIn,
	ImageFormat orderOut
){
	SIZE_CHECK(shapeIn.size() == 4);
	if(orderIn == orderOut){
		output = input;
		return;
	}
	int dimsIn[4] = {int(orderIn)/1000, (int(orderIn) / 100) % 10, (int(orderIn) / 10) % 10, int(orderIn) % 10};
	int dimsOut[4] = {int(orderOut)/1000, (int(orderOut) / 100) % 10, (int(orderOut) / 10) % 10, int(orderOut) % 10};
	
	//compute permutation from input and output
	int dimPerm[4] = {0,0,0,0};
	for(int i = 0; i != 4; ++i){
		while(dimsIn[dimPerm[i]] != dimsOut[i]) ++dimPerm[i];
	}
	//compute dimension strides and sizes
	//those have to be interpreted in: when changing index i_k in output, which stride does that have in input? 
	std::size_t stride[4]={shapeIn.stride(dimPerm[0]) ,shapeIn.stride(dimPerm[1]),shapeIn.stride(dimPerm[2]), shapeIn.stride(dimPerm[3])};
	std::size_t size[4]={shapeIn[dimPerm[0]] ,shapeIn[dimPerm[1]],shapeIn[dimPerm[2]], shapeIn[dimPerm[3]]};
	
	//delegate to device
	reorder_impl(input,output, size, stride);
}

}}

#endif