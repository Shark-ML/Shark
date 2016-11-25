/*!
 *
 *
 * \brief       Implements the default implementation of the POTRF algorithm
 *
 * \author    O. Krause
 * \date        2014
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_POTRF_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_POTRF_HPP

#include "../../expression_types.hpp"
#include "../gemm.hpp"
#include "../trsm.hpp"


namespace shark {namespace blas {namespace bindings {


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
		for(size_t k = j+1; k < m; k++) {
			A()(j,k) = 0.0;
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
	auto transA = trans(A);
	return potrf_block(transA, row_major(), typename Triangular::transposed_orientation());
}

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
	std::size_t split = (A.size1()+block_size-1)/2;
	
	//if the matrix is small enough call the computation kernel directly for the block
	if(size <= block_size){
		return potrf_block(A,typename MatA::orientation(), lower());
	}
	//otherwise run the kernel recursively
	std::size_t result = potrf_recursive(Afull,start,start+split,lower());
	if(result) return result;
	auto Alr = subrange(A,split,size,split,size);
	auto All = trans(subrange(A,split,size,0,split));
	kernels::trsm<false,false>(subrange(A,0,split,0,split),All);
	kernels::gemm(trans(All),All,Alr, -1.0);
	subrange(A,0,split,split,size).clear();
	return potrf_recursive(Afull,start+split,end,lower());
}

template <typename MatA>
std::size_t potrf_recursive(
	matrix_expression<MatA, cpu_tag>& Afull,
	std::size_t start,
	std::size_t end,
	upper
){
	auto transAfull = trans(Afull);
	return potrf_recursive(transAfull,start,end,lower());
}

//dispatcher
template <class Triangular, typename MatA>
std::size_t potrf(
	matrix_container<MatA, cpu_tag>& A,
	boost::mpl::false_//unoptimized
){
	SIZE_CHECK(A().size1() == A().size2());
	return potrf_recursive(A,0,A().size1(), Triangular());
}

}}}
#endif
