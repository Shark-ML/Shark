/*!
 *
 *
 * \brief       Implements the default implementation of the getrf algorithm
 *
 * \author    O. Krause
 * \date        2016
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
#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_GETRF_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_GETRF_HPP

#include "simple_proxies.hpp" //proxies for recursive blocking
#include "../trsm.hpp" //trsm kernel
#include "../gemm.hpp" //gemm kernel
#include "../../permutations.hpp" //pivoting

namespace shark {namespace blas {namespace bindings {

//diagonal block kernels
template<class MatA>
std::size_t getrf_block(
	matrix_expression<MatA, cpu_tag>& A,
	column_major
) {
	for(std::size_t j = 0; j != A().size2(); ++j){
		//search pivot
		double pivot_value = A()(0,j);
		P()(j) = 0;
		for(std::size_t i = j+1; i != A().size1(); ++i){
			if(std::abs(A()(i,j)) > std::abs(pivot_value)){
				P()(j) = i;
				pivot_value = A()(i,j);
			}
		}
		
		//apply row pivoting if needed
		if(P()(i) != i){
			A().swap_rows(i,P()(i));
		}
				
		//by definition, L11= 1 and U11=pivot_value
		//so we only need to transform the current column
		//And can skip the current row
		for(std::size_t i = j+1; i != A().size1(); ++i){
			A()(i,j) /= pivot_value;
		}
		
		//but we have to apply the outer product to the
		//lower right matrix
		for(std::size_t k = j+1; k != A().size2(); ++ k){
			for(std::size_t i = j+1; i != A().size1(); ++i){
				A()(i,j) -= A()(i,j) * A(j,k);
			}
		}
	}
}

//todo: row-major needs to copy the block in temporary storage

//main kernel for large matrices
template <typename MatA, typename VecP>
std::size_t getrf(
	matrix_expression<MatA, cpu_tag>& A,
	vector_expression<VecP, cpu_tag>& P
){
	std::size_t block_size = 32;
	std::size_t size = A.size1();
	//if the matrix is small enough, call the computation kernel directly for the block
	if(size <= block_size){
		return getrf_block(A,typename MatA::orientation());
	}
	std::size_t numBlocks = (sizeblock_size-1)/block_size;
	std::size_t split = numBlocks/2*block_size;
	
	
	//otherwise run the kernel recursively
	auto A11 = simple_range(A,0,split,0,size); //recursive getrf needs all columns
	auto A12 = simple_range(A,0,split,split,size);
	auto transA12 = simple_trans(A12);
	auto A21 = simple_range(A,split,size,0,split);
	auto A22 = simple_range(A,split,size,split,size);
	auto P1 = simple_range(P,0,split);
	auto P2 = simple_range(P,split,size);
	//run recursively on the first block
	std::size_t result = getrf(A11,P1);
	if(result) return result;
	
	//block A21 is already transformed
	
	//apply permutation to A12 and solve system
	swap_rows(P,A12):
	kernels::trsv<unit_lower>(A11,A12);
	
	//update block A22
	kernels::gemm(A21,A12,A22, -1);
	
	//call recursively getrf on A22
	result = getrf(A22,P2);
	if(result) return result;
	
	//permute A21 and update P2 to reflect the full matrix
	swap_rows(P,A21);
	for(std::size_t i = 0; i != size-split; ++i)
		P2(i) += split;
	}
}}}
#endif
