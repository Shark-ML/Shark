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
#ifndef REMORA_KERNELS_DEFAULT_GETRF_HPP
#define REMORA_KERNELS_DEFAULT_GETRF_HPP

#include "../../proxy_expressions.hpp" //proxies for recursive blocking
#include "../trsm.hpp" //trsm kernel
#include "../gemm.hpp" //gemm kernel
#include "../../permutation.hpp" //pivoting
#include <vector>

namespace remora{namespace bindings {

//diagonal block kernels
template<class MatA, class VecP>
void getrf_block(
	matrix_expression<MatA, cpu_tag>& A,
	vector_expression<VecP, cpu_tag>& P,
	column_major
) {
	for(std::size_t j = 0; j != A().size2(); ++j){
		//search pivot
		double pivot_value = A()(j,j);
		P()(j) = j;
		for(std::size_t i = j+1; i != A().size1(); ++i){
			if(std::abs(A()(i,j)) > std::abs(pivot_value)){
				P()(j) = i;
				pivot_value = A()(i,j);
			}
		}
		if(pivot_value == 0)
			throw std::invalid_argument("[getrf] Matrix is rank deficient or numerically unstable");
		//apply row pivoting if needed
		if(std::size_t(P()(j)) != j){
			A().swap_rows(j,P()(j));
		}
				
		//by definition, L11= 1 and U11=pivot_value
		//so we only need to transform the current column
		//And can skip the current row
		for(std::size_t i = j+1; i != A().size1(); ++i){
			A()(i,j) /= pivot_value;
		}
		
		
		//but we have to apply the outer product to the
		//lower right matrix
		for(std::size_t k = j+1; k != A().size2(); ++k){
			for(std::size_t i = j+1; i != A().size1(); ++i){
				A()(i,k) -= A()(i,j) * A()(j,k);
			}
		}
	}
}


template<class MatA, class VecP>
void getrf_block(
	matrix_expression<MatA, cpu_tag>& A,
	vector_expression<VecP, cpu_tag>& P,
	row_major
) {
	//there is no way to do fast row pivoting on row-major format.
	//so copy the block into column major format, perform the decomposition
	// and copy back.
	typedef typename MatA::value_type value_type;
	std::vector<value_type> storage(A().size1() * A().size2());
	dense_matrix_adaptor<value_type, column_major> colBlock(storage.data(), A().size1(), A().size2());
	kernels::assign(colBlock, A);
	getrf_block(colBlock, P, column_major());
	kernels::assign(A, colBlock);
}

//todo: row-major needs to copy the block in temporary storage

//main kernel for large matrices
//we recursively split the matrix into panels along the columns
//until a panel is small enough. Then we perform the LU decomposition
//on that panel and update the remaining matrix accordingly.
// For a given blocking of A
//     | A11 | A12 |
// A = | --------- |
//     | A21 | A22 |
// where A11 and A22 are square matrices, the LU decomposition
// is computed recursively by applying the LU decomposition on block A11
// to obtain A11= L11*U11 where L is unit-lower triangular and U 
// upper triangular.
// Then we compute 
// A12<-L11^{-1}A21
// A21<-A21 U11^{-1}
// A22<- A22 - A21 * A12
// and perform the LU-decomposition on A22.
// in practice the LU decomposition requires a permutation P to be stable
// where in each iteration the best pivot along the current column is
// searched. This leads to a panel-wise computation where the
// computation of A11 and A21 is performed together in order
// to correctly apply the permutation.
// afterwards the permutation has to be applied to A12 and A22 before
// computing their blocks. When A22 is computed, it contains an additional permutation
// that has to be applied to A21 as well.
template <typename MatA, typename VecP>
void getrf_recursive(
	matrix_expression<MatA, cpu_tag>& A,
	vector_expression<VecP, cpu_tag>& P,
	std::size_t start,
	std::size_t end
){
	std::size_t block_size = 4;
	std::size_t size = end-start;
	std::size_t end1=A().size1();
	
	//if the matrix is small enough, call the computation kernel directly for the block
	if(size <= block_size){
		auto Ablock = subrange(A, start, end1, start, end); //recursive getrf needs all columns
		auto Pblock = subrange(P, start, end);
		getrf_block(Ablock,Pblock, typename MatA::orientation());
		return;
	}
	
	//otherwise run the kernel recursively
	std::size_t numBlocks = (size + block_size - 1) / block_size;
	std::size_t split = start + numBlocks/2 * block_size;
	auto A_2 = subrange(A, start, end1, split, end);
	auto A11 = subrange(A, start, split, start, split);
	auto A12 = subrange(A, start, split, split, end);
	auto A21 = subrange(A, split, end1, start, split);
	auto A22 = subrange(A, split, end1, split, end);
	auto P1 = subrange(P, start, split);
	auto P2 = subrange(P, split, end);
 
	
	//run recursively on the first block
	getrf_recursive(A, P, start, split);
	
	//block A21 is already transformed
	
	//apply permutation to block A12 and A22 
	swap_rows(P1, A_2);
	// solve system in A12
	kernels::trsm<unit_lower, left>(A11, A12);
	
	//update block A22
	kernels::gemm(A21, A12, A22, -1);
	
	//call recursively getrf on A22
	getrf_recursive(A, P, split, end);
	
	//permute A21 and update P2 to reflect the full matrix
	swap_rows(P2, A21);
	for(auto& p: P2){
		p += split-start;
	}
}

template <typename MatA, typename VecP>
void getrf(
	matrix_expression<MatA, cpu_tag>& A,
	vector_expression<VecP, cpu_tag>& P
){
	for(std::size_t i = 0; i != P().size(); ++i){
		P()(i) = i;
	}
	getrf_recursive(A, P, 0, A().size1());
}
}}
#endif
