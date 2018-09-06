/*!
 *
 *
 * \brief       Implements the POTRF algorithm for HIP
 *
 * \author    O. Krause
 * \date        2018
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
#ifndef REMORA_KERNELS_HIP_POTRF_HPP
#define REMORA_KERNELS_HIP_POTRF_HPP

#include "../../proxy_expressions.hpp"
#include "../trsm.hpp" //trsm kernel
#include "../syrk.hpp" //syrk kernel

namespace remora{
	
namespace hip{
	
template<size_t TILE_SIZE, class MatA>
__global__ void potrf_block_kernel(hipLaunchParm lp, MatA A, size_t start, size_t end){
	typedef typename std::remove_reference<typename MatA::result_type>::type value_type;
	__shared__ value_type Asub[TILE_SIZE][TILE_SIZE+2];//+2 to avoid bank conflicts
	const size_t numWorkers = hipBlockDim_x;
	//ensure we are not reading out of bounds
	const size_t curTileA = end-start;
	
	// Load tile of A into shared memory
	for(size_t i = hipThreadIdx_x; i < TILE_SIZE; i += numWorkers){
		for(size_t j = hipThreadIdx_y; j < TILE_SIZE; j += numWorkers){
			Asub[i][j] = A(min(end-1, start + i), min(end-1, start + j));
		}
	}
	__threadfence_block();
	// Loop over the values of a single tile
	if(hipThreadIdx_x == 0 && hipThreadIdx_y == 0){
		for(size_t j = 0; j < TILE_SIZE; j++) {
			value_type Ajj = sqrt(Asub[j][j]);
			Asub[j][j] = Ajj;
			for(size_t i = j + 1; i < TILE_SIZE; ++i) {
				Asub[i][j] /= Ajj;
			};
			//rank-one update
			for(size_t k = j + 1; k < TILE_SIZE; k++) {
				for(size_t i = k; i < TILE_SIZE; ++i) {
					Asub[i][k] -= Asub[i][j] * Asub[k][j];
				}
			}
		}
	}
	// Synchronise before continuing
	__threadfence_block();
	// Store the final results back in A
	for(size_t i = hipThreadIdx_x; i < curTileA; i += numWorkers){
		for(size_t j = hipThreadIdx_y; j < curTileA; j += numWorkers){
			A(start + i, start+j) = Asub[i][j];
		}
	}
}

//main kernel for large matrices
template <typename MatA>
void potrf_recursive(
	matrix_expression<MatA, hip_tag>& Afull,
	std::size_t start,
	std::size_t end,
	lower
){
	const std::size_t block_size = 16;
	auto A = subrange(Afull,start,end,start,end);
	std::size_t size = A.size1();
	//if the matrix is small enough call the computation kernel directly for the block
	if(size <= block_size){
		std::size_t blockSize = std::size_t(std::sqrt(double(Afull().queue().warp_size())));
		auto stream = hip::get_stream(Afull().queue()).handle();
		hipLaunchKernel(
			hip::potrf_block_kernel<block_size>, 
			dim3(1,1), dim3(blockSize, blockSize), 0, stream, 
			Afull().elements(), start, end
		);
		return;
	}
	std::size_t numBlocks = (A.size1()+block_size-1)/block_size;
	std::size_t split = numBlocks/2*block_size;
	
	//otherwise run the kernel recursively
	potrf_recursive(Afull,start,start+split, lower());
	
	auto Aul = subrange(A,0,split,0,split);
	auto All = subrange(A,split,size,0,split);
	auto Alr = subrange(A,split,size,split,size);
	kernels::trsm<upper,right>(trans(Aul), All );
	kernels::syrk<false>(All,Alr, -1.0);
	potrf_recursive(Afull,start+split,end, lower());
}

template <typename MatA>
void potrf_recursive(
	matrix_expression<MatA, hip_tag>& A,
	std::size_t start,
	std::size_t end,
	upper
){
	auto Atrans = trans(A);
	potrf_recursive(Atrans, start, end, lower());
}

}

namespace kernels {
template <class Triangular, typename MatA>
std::size_t potrf(
	matrix_container<MatA, hip_tag>& A
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	A().queue().set_device();
	hip::potrf_recursive(A, 0, A().size1(), Triangular());
	return 0;
}

}}
#endif
