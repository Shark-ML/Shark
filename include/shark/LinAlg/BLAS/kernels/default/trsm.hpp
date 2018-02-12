/*!
 *
 *
 * \brief       -
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

#ifndef REMORA_KERNELS_DEFAULT_TRSM_HPP
#define REMORA_KERNELS_DEFAULT_TRSM_HPP

#include "../../expression_types.hpp" //matrix_expression
#include "../../proxy_expressions.hpp" //matrix proxies for blocking
#include "../../detail/structure.hpp" //structure tags
#include "../gemm.hpp" //gemm kernel

#include <stdexcept> //exception when matrix is singular
#include <type_traits> //std::false_type marker for unoptimized

namespace remora{namespace bindings {

//Lower triangular - matrix(row-major)
template<std::size_t maxBlockSize1,std::size_t maxBlockSize2, bool Unit, class MatA, class MatB>
void trsm_block(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag> &B,
	lower,
	row_major // B is row-major
){
	REMORA_SIZE_CHECK(A().size1() <= maxBlockSize1);
	typedef typename MatA::value_type value_typeA;
	typedef typename MatB::value_type value_typeB;


	//evaluate and copy block of A
	std::size_t size = A().size1();
	value_typeA blockA[maxBlockSize1][maxBlockSize1];
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j <= i; ++j){
			blockA[i][j] = A()(i,j);
		}
	}


	value_typeB blockB[maxBlockSize2][maxBlockSize1];
	std::size_t numBlocks = (B().size2()+maxBlockSize2-1)/maxBlockSize2;
	for(std::size_t i = 0; i != numBlocks; ++i){
		std::size_t startB= i*maxBlockSize2;
		std::size_t curBlockSize2 =std::min(maxBlockSize2, B().size2() - startB);

		//copy blockB transposed in memory
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t k = 0; k != curBlockSize2; ++k){
				blockB[k][i] = B()(i,startB+k);
			}
		}
		//compute trsv kernel for each row in blockB
		for(std::size_t k = 0; k != curBlockSize2; ++k){
			for (std::size_t i = 0; i != size; ++i) {
				for (std::size_t j = 0; j != i; ++j) {
					blockB[k][i] -= blockA[i][j]*blockB[k][j];
				}
				if(!Unit){
					if(blockA[i][i] == value_typeA())
						throw std::invalid_argument("[TRSM] Matrix is singular!");
					blockB[k][i] /= blockA[i][i];
				}
			}
		}
		//copy blockB back
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t k = 0; k != curBlockSize2; ++k){
				B()(i,startB+k) = blockB[k][i];
			}
		}
	}
}

// Lower triangular - matrix(column-major)
template<std::size_t maxBlockSize1,std::size_t maxBlockSize2, bool Unit, class MatA, class MatB>
void trsm_block(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag>& B,
	lower,
	column_major // B is column-major
) {
	typedef typename MatA::value_type value_type;
	//evaluate and copy block of A
	std::size_t size = A().size1();
	value_type blockA[maxBlockSize1][maxBlockSize1];
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = 0; j <= i; ++j){
			blockA[i][j] = A()(i,j);
		}
	}

	//compute trsv kernel for each column in B
	for(std::size_t k = 0; k != B().size2(); ++k){
		for (std::size_t i = 0; i != size; ++i) {
			for (std::size_t j = 0; j != i; ++j) {
				B()(i,k) -= blockA[i][j] * B()(j,k);
			}
			if(!Unit){
				if(blockA[i][i] == value_type())
					throw std::invalid_argument("[TRSM] Matrix is singular!");
				B()(i,k) /= blockA[i][i];
			}
		}
	}
}


//Upper triangular - matrix(row-major)
template<std::size_t maxBlockSize1, std::size_t maxBlockSize2, bool Unit, class MatA, class MatB>
void trsm_block(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag> &B,
    upper,
	row_major // B is row-major
){
	REMORA_SIZE_CHECK(A().size1() <= maxBlockSize1);
	typedef typename MatA::value_type value_typeA;
	typedef typename MatB::value_type value_typeB;

	//evaluate and copy block of A
	std::size_t size = A().size1();
	value_typeA blockA[maxBlockSize1][maxBlockSize1];
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = i; j != size; ++j){
			blockA[i][j] = A()(i,j);
		}
	}

	value_typeB blockB[maxBlockSize2][maxBlockSize1];
	std::size_t numBlocks = (B().size2()+maxBlockSize2-1)/maxBlockSize2;
	for(std::size_t i = 0; i != numBlocks; ++i){
		std::size_t startB= i*maxBlockSize2;
		std::size_t curBlockSize2 =std::min(maxBlockSize2, B().size2() - startB);

		//copy blockB transposed in memory
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t k = 0; k != curBlockSize2; ++k){
				blockB[k][i] = B()(i,startB+k);
			}
		}
		//compute trsv kernel for each row in blockB
		for(std::size_t k = 0; k != curBlockSize2; ++k){
			for (std::size_t n = 0; n != size; ++n) {
				std::size_t i = size-n-1;
				for (std::size_t j = i+1; j != size; ++j) {
					blockB[k][i] -= blockA[i][j] * blockB[k][j];
				}
				if(!Unit){
					if(blockA[i][i] == value_typeA())
						throw std::invalid_argument("[TRSM] Matrix is singular!");
					blockB[k][i] /= blockA[i][i];
				}
			}
		}
		//copy blockB back
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t j = 0; j != curBlockSize2; ++j){
				B()(i,startB+j) = blockB[j][i];
			}
		}
	}
}

// Upper triangular - matrix(column-major)
template<std::size_t maxBlockSize1,std::size_t maxBlockSize2, bool Unit, class MatA, class MatB>
void trsm_block(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag>& B,
	upper,
	column_major // B is column-major
) {
	typedef typename MatA::value_type value_type;
	//evaluate and copy block of A
	std::size_t size = A().size1();
	value_type blockA[maxBlockSize1][maxBlockSize1];
	for(std::size_t i = 0; i != size; ++i){
		for(std::size_t j = i; j != size; ++j){
			blockA[i][j] = A()(i,j);
		}
	}

	//compute trsv kernel for each column in B
	for(std::size_t k = 0; k != B().size2(); ++k){
		for (std::size_t n = 0; n != size; ++n) {
			std::size_t i = size-n-1;
			for (std::size_t j = i+1; j != size; ++j) {
				B()(i,k) -= blockA[i][j] * B()(j,k);
			}
			if(!Unit){
				if(blockA[i][i] == value_type())
					throw std::invalid_argument("[TRSM] Matrix is singular!");
				B()(i,k) /= blockA[i][i];
			}
		}
	}
}

template <typename MatA, typename MatB, class Triangular>
void trsm_recursive(
	matrix_expression<MatA, cpu_tag> const& Afull,
	matrix_expression<MatB, cpu_tag> & Bfull,
	std::size_t start,
	std::size_t end,
	Triangular t,
	left l
){
	static std::size_t const Block_Size =  32;
	std::size_t num_rhs = Bfull().size2();
	auto A = subrange(Afull,start,end,start,end);
	auto B = subrange(Bfull,start,end,0,num_rhs);
	//if the matrix is small enough call the computation kernel directly for the block
	if(A.size1() <= Block_Size){
		trsm_block<Block_Size,16,Triangular::is_unit>(A,B,triangular_tag<Triangular::is_upper,false>(), typename MatB::orientation());
		return;
	}
	std::size_t size = A.size1();
	std::size_t numBlocks =(A.size1()+Block_Size-1)/Block_Size;
	std::size_t split = numBlocks/2*Block_Size;
	auto Bfront = subrange(B,0,split,0,num_rhs);
	auto Bback = subrange(B,split,size,0,num_rhs);

	//otherwise run the kernel recursively
	if(Triangular::is_upper){ //Upper triangular case
		trsm_recursive(Afull, Bfull,start+split,end, t, l);
		kernels::gemm(subrange(A,0,split,split,size), Bback, Bfront, -1.0);
		trsm_recursive(Afull, Bfull,start,start+split, t, l);
	}else{// Lower triangular caste
		trsm_recursive(Afull, Bfull,start,start+split, t, l);
		kernels::gemm(subrange(A,split,size,0,split), Bfront, Bback, -1.0);
		trsm_recursive(Afull, Bfull,start+split,end, t, l);
	}
}

template <typename MatA, typename MatB, class Triangular>
void trsm_recursive(
	matrix_expression<MatA, cpu_tag> const& Afull,
	matrix_expression<MatB, cpu_tag> & Bfull,
	std::size_t start,
	std::size_t end,
	Triangular,
	right
){
	auto transA = trans(Afull);
	auto transB = trans(Bfull);
	trsm_recursive(transA,transB,start,end,typename Triangular::transposed_orientation(),left());
}

//main kernel runs the kernel above recursively and calls gemv
template <class Triangular, class Side, typename MatA, typename MatB>
void trsm(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag>& B,
	std::false_type //unoptimized
){

	bindings::trsm_recursive(A,B,0,A().size1(), Triangular(), Side());
}
}}
#endif
