/*!
 * \brief       kernels for getting the maximum element of a vector with hip
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
#ifndef REMORA_KERNELS_HIP_VECTOR_MAX_HPP
#define REMORA_KERNELS_HIP_VECTOR_MAX_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include "../../hip/buffer.hpp"
namespace remora{
namespace hip{
template<class VecV>
__global__ void vector_max_kernel(hipLaunchParm lp, VecV v, size_t size, size_t* max){
	typedef typename std::remove_const<
		typename std::remove_reference<typename VecV::result_type>::type 
	> ::type value_type;
	__shared__ value_type max_value[64];
	__shared__ std::size_t max_index[64];
	value_type& thread_max = max_value[hipThreadIdx_x];
	std::size_t& thread_index = max_index[hipThreadIdx_x];
	thread_max = 1.e-30;
	thread_index = 0;
	for(size_t i = hipThreadIdx_x; i < size; i += hipBlockDim_x){
		if(thread_max < v(i)){
			thread_max = v(i);
			thread_index = i;
		}
	}
	__threadfence();
	
	if(hipThreadIdx_x == 0){
		for(size_t i = 1 ; i < min(size_t(hipBlockDim_x), size); ++i){
			if(thread_max < max_value[i]){
				thread_max = max_value[i];
				thread_index = max_index[i];
			}
		}
		*max = thread_index;
	}
}
}
namespace bindings{
template<class V>
std::size_t vector_max(vector_expression<V, hip_tag> const& v, dense_tag){
	if(v().size() == 0) return 0;
	hip::buffer<std::size_t> result(1, v().queue());
	
	std::size_t blockSize = std::min(64, v().queue().warp_size());
	std::size_t numBlocks = (v().size() + blockSize - 1) / blockSize;
	auto stream = hip::get_stream(v().queue()).handle();
	hipLaunchKernel(
		hip::vector_max_kernel, 
		dim3(numBlocks), dim3(blockSize), 0, stream,
		v().elements(), v().size(), result.get()
	);
	std::size_t index;
	hipMemcpy(&index, result.get(), sizeof(index), hipMemcpyDeviceToHost);
	return index;
}


}}
#endif
