/*!
 *  \brief Cholesky Decompositions for a positive semi-definite Matrix A = LL^T
 *
 *
 *  \author  O. Krause
 *  \date    2016
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

#ifndef REMORA_KERNELS_DEFAULT_PSTRF_HPP
#define REMORA_KERNELS_DEFAULT_PSTRF_HPP

#include "../gemm.hpp" //gemm kernel
#include "../gemv.hpp" //gemv kernel
#include "../../proxy_expressions.hpp"
#include "../../dense.hpp"
#include <algorithm>
namespace remora{namespace bindings {

template<class MatA, class VecP>
std::size_t pstrf(
	matrix_expression<MatA, cpu_tag> &A,
	vector_expression<VecP, cpu_tag>& P,
	lower
){
	//The normal cholesky decomposition works by
	// partitioning A in 4 blocks. 
	//      |A11 | A12
	//A^(k)=|-----------
	//      |A21 | A^(k+1)
	// First, the cholesky decomposition of A is computed, after which 
	// bloc A12 is computed by solving a system of equations.
	// Lastly, the (expensive) operation
	// A^(k+1) -= A21 A21^T
	// is performed. 
	// When we suspect that A is rank deficient, this does not work
	// as we might run in the case that A11 is rank deficient which makes
	// updating A21 impossible.
	// instead, in every iteration we first pick the row/column with the
	// current largest diagonal element, permute the matrix, so that
	// this row/column is in the blocks A11 and A12 and update everything
	//step by step until we have a full block, which makes it possible
	// to perform the expensive 
	// // A^(k+1) -= A21 A21^T
	// using efficient routines.
	
	
	//todo: experiment a bit with the sizes
	std::size_t block_size = 20;
	
	
	size_t m = A().size1();
	//storage for pivot values
	vector<typename MatA::value_type> pivotValues(m);
	
	//stopping criterion
	double max_diag = A()(0,0);
	for(std::size_t i = 1; i < m; ++i)
		max_diag = std::max(max_diag,std::abs(A()(i,i)));
	double epsilon = m * m * std::numeric_limits<typename MatA::value_type>::epsilon() * max_diag;
	
	for(std::size_t k = 0; k < m; k += block_size){
		std::size_t currentSize = std::min(m-k,block_size);//last iteration is smaller
		//partition of the matrix
		auto Ak = subrange(A,k,m,k,m);
		auto pivots = subrange(pivotValues,k,m);
		
		//update current block L11
		for(std::size_t j = 0; j != currentSize; ++j){
			//we have to dynamically update the pivot values
			//we start every block anew to prevent accumulation of rounding errors
			if(j == 0){
				for(std::size_t i = 0; i != m-k; ++i)
					pivots(i) = Ak(i,i);
			}else{//update pivot values
				for(std::size_t i = j; i != m-k; ++i)
					pivots(i) -= Ak(i,j-1) * Ak(i,j-1);
			}
			//get current pivot. if it is not equal to the j-th, we swap rows and columns
			//such that j == pivot and Ak(j,j) = pivots(pivot)
			std::size_t pivot = std::max_element(pivots.begin()+j,pivots.end())-pivots.begin();
			if(pivot != j){
				P()(k+j) = (int)(pivot+k);
				A().swap_rows(k+j,k+pivot);
				A().swap_columns(k+j,k+pivot);
				std::swap(pivots(j),pivots(pivot));
			}
			
			//check whether we are finished
			auto pivotValue = pivots(j);
			if(pivotValue < epsilon){
				//the remaining part is so near 0, we can just ignore it
				subrange(Ak,j,m-k,j,m-k).clear();
				return k+j;
			}
			
			//update current column
			Ak(j,j) = std::sqrt(pivotValue);
			//the last updates of columns k...k+j-1 did not change
			//this row, so do it now
			auto colLowerPart = subrange(column(Ak,j),j+1,m-k);
			if(j > 0){
				//the last updates of columns 0,1,...,j-1 did not change
				//this column, so do it now
				auto blockLL = subrange(Ak,j+1,m-k,0,j);
				auto curRow = row(Ak,j);
				auto rowLeftPart = subrange(curRow,0,j);
				
				//suppose you are the j-th column
				//than you want to get updated by the last
				//outer products which are induced by your column friends
				//Li...Lj-1
				//so you get the effect of
				//(L1L1T)^(j)+...+(Lj-1Lj-1)^(j)
				//=L1*L1j+L2*L2j+L3*L3j...
				//which is a matrix-vector product
				kernels::gemv(blockLL,rowLeftPart,colLowerPart,-1);
			}
			colLowerPart /= Ak(j,j);
			//set block L12 to 0
			subrange(Ak,j,j+1,j+1,Ak.size2()).clear();
		}
		if(k+currentSize < m){
			auto blockLL = subrange(Ak, block_size, m-k, 0, block_size);
			auto blockLR = subrange(Ak, block_size, m-k,  block_size, m-k);
			kernels::gemm(blockLL,trans(blockLL), blockLR, -1);
		}
	}
	return m;
	
}

template<class MatA, class VecP>
std::size_t pstrf(
	matrix_expression<MatA, cpu_tag> &A,
	vector_expression<VecP, cpu_tag>& P,
	upper
){
	auto transA = trans(A);
	return pstrf(transA,P,lower());
}

}}
#endif