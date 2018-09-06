/*!
 * \brief       Assignment kernels for vector expressions
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
#ifndef REMORA_KERNELS_HIP_VECTOR_ASSIGN_HPP
#define REMORA_KERNELS_HIP_VECTOR_ASSIGN_HPP

#include "../../expression_types.hpp"
#include "../../hip/traits.hpp"

namespace remora{
namespace hip{
template<class V, class F>
__global__ void vector_apply_kernel(hipLaunchParm lp, V v, size_t size, F f){
	std::size_t i = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
	if(i < size)
		v(i) = f(v(i));
}

template<class V,  class E, class F>
__global__ void vector_assign_functor_kernel(hipLaunchParm lp, V v, size_t size, E e, F f){
	std::size_t i = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
	if(i < size)
		v(i) = f(v(i), e(i));
}

}
	
	
namespace bindings{

template<class F, class V>
void apply(vector_expression<V, hip_tag>& v, F const& f){
	v().queue().set_device();
	std::size_t blockSize = v().queue().warp_size();
	std::size_t numBlocks = (v().size() + blockSize - 1) / blockSize;
	auto stream = hip::get_stream(v().queue()).handle();
	hipLaunchKernel(
		hip::vector_apply_kernel, 
		dim3(numBlocks), dim3(blockSize), 0, stream, 
		v().elements(), v().size(), f
	);
}

template<class F, class V>
void assign(vector_expression<V, hip_tag>& v, typename V::value_type t) {
	static_assert(std::is_base_of<dense_tag, typename V::storage_type::storage_tag>::value, "target must have dense storage for assignment");
	auto f = device_traits<hip_tag>::make_bind_second(F(), t);
	apply(v,f);
}

////////////////////////////////////////////
//assignment with functor
////////////////////////////////////////////

// Dense-Dense case
template<class V, class E, class F>
void vector_assign_functor(
	vector_expression<V, hip_tag>& v,
	vector_expression<E, hip_tag> const& e,
	F f,
	dense_tag, dense_tag
) {
	v().queue().set_device();
	std::size_t blockSize = v().queue().warp_size();
	std::size_t numBlocks = (v().size() + blockSize - 1) / blockSize;
	auto stream = hip::get_stream(v().queue()).handle();
	hipLaunchKernel(
		hip::vector_assign_functor_kernel, 
		dim3(numBlocks), dim3(blockSize), 0, stream,
		v().elements(), v().size(), e().elements(), f
	);
}

/////////////////////////////////////////////////////////
//direct assignment of two vectors
////////////////////////////////////////////////////////

// Dense-Dense case
template< class V, class E>
void vector_assign(
	vector_expression<V, hip_tag>& v, vector_expression<E, hip_tag> const& e, 
	dense_tag t, dense_tag
) {
	vector_assign_functor(v, e, device_traits<hip_tag>::right_arg<typename E::value_type>(), t, t);
}




}}
#endif
