/*!
 *
 *
 * \brief       Implements the default implementation of the POTRF algorithm
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
#ifndef REMORA_KERNELS_DEFAULT_POTRF_HPP
#define REMORA_KERNELS_DEFAULT_POTRF_HPP

#include "../../proxy_expressions.hpp"
#include "../trsm.hpp" //trsm kernel
#include "../syrk.hpp" //syrk kernel
#include <type_traits> //std::false_type marker for unoptimized

namespace remora{namespace bindings {

//diagonal block kernels
//upper potrf(row-major)
template<class MatA>
std::size_t potrf_block(
	matrix_expression<MatA, cpu_tag>& A,
	row_major, lower
) {
	std::size_t m = A().size1();
	for(size_t j = 0; j < m; j++) {
		for(size_t i = j; i < m; i++) {
			double s = A()(i, j);
			for(size_t k = 0; k < j; k++) {
				s -= A()(i, k) * A()(j, k);
			}
			if(i == j) {
				if(s <= 0)
					return i+1;
				A()(i, j) = std::sqrt(s);
			} else {
				A()(i, j) = s / A()(j , j);
			}
		}
	}
	return 0;
}

//lower potrf(row-major)
template<class MatA>
std::size_t potrf_block(
	matrix_expression<MatA, cpu_tag>& A,
	row_major, upper
) {
	std::size_t m = A().size1();
	for(size_t i = 0; i < m; i++) {
		double& Aii = A()(i, i);
		if(Aii < 0)
			return i+1;
		using std::sqrt;
		Aii = sqrt(Aii);
		//update row
		
		for(std::size_t j = i + 1; j < m; ++j) {
			A()(i, j) /= Aii;
		}
		//rank-one update
		for(size_t k = i + 1; k < m; k++) {
			for(std::size_t j = k; j < m; ++j) {
				A()(k, j) -= A()(i, k) * A()(i, j);
			}
		}
	}
	return 0;
}


//dispatcher for column major
template<class MatA, class Triangular>
std::size_t potrf_block(
    matrix_expression<MatA, cpu_tag>& A,
    column_major, Triangular
) {
	auto Atrans = trans(A);
	return potrf_block(Atrans, row_major(), typename Triangular::transposed_orientation());
}

//main kernel for large matrices
template <typename MatA>
std::size_t potrf_recursive(
	matrix_expression<MatA, cpu_tag>& Afull,
	std::size_t start,
	std::size_t end,
	lower
){
	std::size_t block_size = 32;
	auto A = subrange(Afull,start,end,start,end);
	std::size_t size = A.size1();
	//if the matrix is small enough call the computation kernel directly for the block
	if(size <= block_size){
		return potrf_block(A,typename MatA::orientation(), lower());
	}
	std::size_t numBlocks = (A.size1()+block_size-1)/block_size;
	std::size_t split = numBlocks/2*block_size;
	
	
	//otherwise run the kernel recursively
	std::size_t result = potrf_recursive(Afull,start,start+split,lower());
	if(result) return result;
	
	auto Aul = subrange(A,0,split,0,split);
	auto All = subrange(A,split,size,0,split);
	auto Alr = subrange(A,split,size,split,size);
	kernels::trsm<upper,right>(trans(Aul), All );
	kernels::syrk<false>(All,Alr, -1.0);
	return potrf_recursive(Afull,start+split,end,lower());
}

template <typename MatA>
std::size_t potrf_recursive(
	matrix_expression<MatA, cpu_tag>& A,
	std::size_t start,
	std::size_t end,
	upper
){
	auto Atrans = trans(A);
	return potrf_recursive(Atrans,start,end,lower());
}

//dispatcher
template <class Triangular, typename MatA>
std::size_t potrf(
	matrix_container<MatA, cpu_tag>& A,
	std::false_type//unoptimized
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	return potrf_recursive(A,0,A().size1(), Triangular());
}

}}
#endif
