/*!
 * \brief       Kernels for matrix-expression assignments on hip
 * 
 * \author      O. Krause
 * \date        2018
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
#ifndef REMORA_KERNELS_HIP_MATRIX_ASSIGN_HPP
#define REMORA_KERNELS_HIP_MATRIX_ASSIGN_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"

#include <iostream>
namespace remora{

namespace hip{
template<class M, class F>
__global__ void matrix_apply_kernel(hipLaunchParm lp, M m, size_t size1, size_t size2, F f){
	size_t row_start = (hipBlockIdx_x * 16);
	size_t column_start = (hipBlockIdx_y * 16);
	size_t row_end = min(row_start + 16, size1);
	size_t column_end = min(column_start + 16, size2);
	for(size_t i = row_start+ hipThreadIdx_x; i < row_end; i += hipBlockDim_x){
		for(size_t j = column_start+ hipThreadIdx_y; j < column_end; j += hipBlockDim_y){
			m(i, j) = f(m(i, j));
		}
	}
}

//kernel M row-major, E row-major
template<class M, class E, class F>
__global__ void matrix_assign_kernel_rowE(hipLaunchParm lp, M m, size_t size1, size_t size2, E e, F f){
	size_t base_row = (hipBlockIdx_x * 16);
	size_t base_column = (hipBlockIdx_y * 16);
	size_t block_size1 = min(size_t(16), size1 - base_row);
	size_t block_size2 = min(size_t(16), size2 - base_column);
	for(size_t i = hipThreadIdx_x; i < block_size1; i += hipBlockDim_x){
		for(size_t j = hipThreadIdx_y; j < block_size2; j += hipBlockDim_y){
			m(base_row + i, base_column + j) = f(m(base_row + i, base_column + j), e(base_row + i, base_column + j));
		}
	}
}
//kernel M row-major, E column-major
template<class M, class E, class F>
__global__ void matrix_assign_kernel_colE(hipLaunchParm lp, M m, size_t size1, size_t size2, E e, F f){
	typedef typename std::remove_reference<typename M::result_type>::type value_type;
	size_t base_row = (hipBlockIdx_x * 16);
	size_t base_column = (hipBlockIdx_y * 16);
	size_t block_size1 = min(size_t(16), size1 - base_row);
	size_t block_size2 = min(size_t(16), size2 - base_column);
	__shared__ value_type block[16][16 + 1];
	for(size_t j = hipThreadIdx_y; j < block_size2; j += hipBlockDim_y){
		for(size_t i = hipThreadIdx_x; i < block_size1; i += hipBlockDim_x){
			block[i][j] = e(base_row + i, base_column + j);
		}
	}
	__threadfence_block();
	for(size_t i = hipThreadIdx_x; i < block_size1; i += hipBlockDim_x){
		for(size_t j = hipThreadIdx_y; j < block_size2; j += hipBlockDim_y){
			value_type& val = m(base_row + i, base_column + j);
			val = f(val, block[i][j]);
		}
	}
}




}
	
	
namespace bindings{
	
	
//////////////////////////////////////////////////////
////Apply function elementwise to Matrix
/////////////////////////////////////////////////////

// Explicitly iterating row major
template<class F, class M, class Orientation>
void matrix_apply(
	matrix_expression<M, hip_tag>& m, 
	F const& f,
	Orientation
){
	//matrix is tiled in 16x16 blocks the block size are sub-blocks in this matrix
	std::size_t blockSize1 = std::min(16, m().queue().warp_size());
	std::size_t blockSize2 = std::min<std::size_t>(16,m().queue().warp_size() / blockSize1);
	std::size_t numBlocks1 = (m().size1() + 16 - 1) / 16;
	std::size_t numBlocks2 = (m().size2()  + 16 - 1) / 16;
	auto stream = get_stream(m().queue()).handle();
	hipLaunchKernel(
		hip::matrix_apply_kernel, 
		dim3(numBlocks1, numBlocks2), dim3(blockSize1, blockSize2), 0, stream,
		m().elements(), m().size1(), m().size2(), f
	);
}
	
//////////////////////////////////////////////////////
////Scalar Assignment to Matrix
/////////////////////////////////////////////////////

template<class F, class M, class Orientation>
void matrix_assign(
	matrix_expression<M, hip_tag>& m, 
	typename M::value_type t, 
	Orientation o
){
	static_assert(std::is_base_of<dense_tag, typename M::storage_type::storage_tag>::value, "target must have dense storage for assignment");
	auto f = device_traits<hip_tag>::make_bind_second(F(), t);
	matrix_apply(m,f,o);
}


///////////////////////////////////////////////////////////////////////////////////////////
//////Matrix Assignment With Functor implementing +=,-=...
///////////////////////////////////////////////////////////////////////////////////////////

template<class F, class M, class E>
void matrix_assign_functor(
	matrix_expression<M, hip_tag>& m, 
	matrix_expression<E, hip_tag> const& e,
	F f,
	row_major, row_major ,dense_tag, dense_tag
){
	//matrix is tiled in 16x16 blocks the block size are sub-blocks in this matrix
	std::size_t blockSize1 = std::min(16, m().queue().warp_size());
	std::size_t blockSize2 = std::min<std::size_t>(16,m().queue().warp_size() / blockSize1);
	std::size_t numBlocks1 = (m().size1() + 16 - 1) / 16;
	std::size_t numBlocks2 = (m().size2() + 16 - 1) / 16;
	auto stream = get_stream(m().queue()).handle();
	hipLaunchKernel(
		hip::matrix_assign_kernel_rowE, 
		dim3(numBlocks1, numBlocks2), dim3(blockSize1, blockSize2), 0, stream,
		m().elements(), m().size1(), m().size2(), 
		e().elements(), f
	);
}

//dense-dense case row-major, column-major
template<class F,class M, class E>
void matrix_assign_functor(
	matrix_expression<M, hip_tag>& m, 
	matrix_expression<E, hip_tag> const& e,
	F f,
	row_major, column_major ,dense_tag, dense_tag
) {
	//matrix is tiled in 16x16 blocks the block size are sub-blocks in this matrix
	std::size_t blockSize2 = std::min<std::size_t>(16, m().queue().warp_size());
	std::size_t blockSize1 = std::min<std::size_t>(16,m().queue().warp_size() / blockSize2);
	std::size_t numBlocks1 = (m().size1() + 16 - 1) / 16;
	std::size_t numBlocks2 = (m().size2() + 16 - 1) / 16;
	auto stream = get_stream(m().queue()).handle();
	hipLaunchKernel(
		hip::matrix_assign_kernel_colE, 
		dim3(numBlocks1, numBlocks2), dim3(blockSize1, blockSize2), 0, stream,
		m().elements(), m().size1(), m().size2(), 
		e().elements(), f
	);
}

/////////////////////////////////////////////////////////////////
//////Matrix Assignment implementing op=
////////////////////////////////////////////////////////////////

template<class M, class E>
void matrix_assign(
	matrix_expression<M, hip_tag> &m, 
	matrix_expression<E, hip_tag> const& e,
	row_major o, row_major,dense_tag t, dense_tag
) {
	matrix_assign_functor(m, e, device_traits<hip_tag>::right_arg<typename E::value_type>(), o, o, t, t);
}

//dense-dense case
template<class M, class E>
void matrix_assign(
	matrix_expression<M, hip_tag> &m, 
	matrix_expression<E, hip_tag> const& e,
	row_major o1, column_major o2,dense_tag t, dense_tag
) {
	matrix_assign_functor(m, e, device_traits<hip_tag>::right_arg<typename E::value_type>(), o1, o2, t, t);
}


}}

#endif
