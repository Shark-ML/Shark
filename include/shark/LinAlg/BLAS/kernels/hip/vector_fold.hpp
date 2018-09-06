/*!
 * \brief       kernels for folding vectors with hip
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
#ifndef REMORA_KERNELS_HIP_VECTOR_FOLD_HPP
#define REMORA_KERNELS_HIP_VECTOR_FOLD_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include "../../hip/buffer.hpp"
namespace remora{
namespace hip{
template<class VecV, class R, class F>
__global__ void vector_fold_kernel(hipLaunchParm lp, VecV v, size_t size, R* resultp, F f){
	__shared__ R folds[64];
	R& entry = folds[hipThreadIdx_x];
	size_t i = hipThreadIdx_x;
	if(i < size){
		entry = v(i);
		i += hipBlockDim_x;
		for(;i < size; i +=  hipBlockDim_x){
			entry = f(entry, v(i));
		}
	}
	__threadfence();
	
	if(hipThreadIdx_x == 0){
		for(size_t i = 0 ; i < min(size_t(hipBlockDim_x), size); ++i){
			*resultp = f(*resultp, folds[i]);
		}
	}
}
}
namespace bindings{
template<class F, class V>
void vector_fold(vector_expression<V, hip_tag> const& v, typename F::result_type& value, dense_tag){
	if(v().size() == 0) return;
	typedef typename F::result_type value_type;
	hip::buffer<value_type> result(1, v().queue());
	
	hipMemcpy(result.get(), &value, sizeof(value), hipMemcpyHostToDevice);
	
	std::size_t blockSize = std::min(64, v().queue().warp_size());
	std::size_t numBlocks = 1;
	auto stream = hip::get_stream(v().queue()).handle();
	hipLaunchKernel(
		hip::vector_fold_kernel, 
		dim3(numBlocks), dim3(blockSize), 0, stream,
		v().elements(), v().size(), result.get(), F()
	);
	hipMemcpy(&value, result.get(), sizeof(value), hipMemcpyDeviceToHost);
}


}}
#endif
