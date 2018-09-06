/*!
 * 
 *
 * \brief       Folds the rows of a row-major or column major matrix.
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

#ifndef REMORA_KERNELS_HIP_FOLD_ROWS_HPP
#define REMORA_KERNELS_HIP_FOLD_ROWS_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"

namespace remora{

namespace hip{
template<class MatA, class VecV, class F, class G>
__global__ void fold_rows_kernel(hipLaunchParm lp,MatA A, size_t size1, size_t size2, VecV v, F f, G g){
	typedef typename std::remove_reference<typename VecV::result_type>::type value_type;
	__shared__ value_type folds[64];
	size_t rowid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	size_t colid = hipThreadIdx_y;
	value_type& entry = folds[hipThreadIdx_y];
	if(colid < size2){
		entry = A(rowid,colid);
		colid += hipBlockDim_y;
		for(;colid < size2; colid += hipBlockDim_y){
			entry = f(entry, A(rowid,colid));
		}
	}
	__threadfence_block();
	if(hipThreadIdx_y == 0){
		value_type acc =  folds[0];
		for(size_t i = 1 ; i < min(size_t(hipBlockDim_y), size2); ++i){
			acc = f(acc, folds[i]);
		}
		v(rowid) += g(acc);
	}
}
}
	
namespace bindings{

template<class F, class G, class MatA, class VecV, class Orientation>
void fold_rows(
	matrix_expression<MatA, hip_tag> const& A, 
	vector_expression<VecV, hip_tag>& v,
	F f,
	G g,
	Orientation
){
	std::size_t blockSize1 = 1;
	std::size_t blockSize2 = std::min<std::size_t>(64, A().queue().warp_size());
	std::size_t numBlocks1 = A().size1();
	std::size_t numBlocks2 = 1;
	auto stream = get_stream(A().queue()).handle();
	hipLaunchKernel(
		hip::fold_rows_kernel, 
		dim3(numBlocks1, numBlocks2), dim3(blockSize1, blockSize2), 0, stream,
		A().elements(), A().size1(), A().size2(),
		v().elements(), f, g
	);
}


}}

#endif
