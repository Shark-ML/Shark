/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2010
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

#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_TRMM_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_TRMM_HPP

#include <boost/mpl/bool.hpp>
#include "../gemm.hpp"

namespace shark { namespace blas { namespace bindings {
	

//Lower triangular - matrix(row-major)
template<std::size_t maxBlockSize1,std::size_t maxBlockSize2, bool Unit, class MatA, class MatB>
void trmm_block(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag> &B,
        boost::mpl::false_, //Lower
	row_major // B is row-major
){
	SIZE_CHECK(A().size1() <= maxBlockSize1);
	typedef typename MatB::value_type value_typeB;

	value_typeB block[maxBlockSize2][maxBlockSize1];
	std::size_t numBlocks = (B().size2()+maxBlockSize2-1)/maxBlockSize2;
	std::size_t size = A().size1();
	for(std::size_t i = 0; i != numBlocks; ++i){
		std::size_t startB= i*maxBlockSize2;
		std::size_t curBlockSize2 =std::min(maxBlockSize2, B().size2() - startB);
		
		//copy block transposed in memory
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t j = 0; j != curBlockSize2; ++j){
				block[j][i] = B()(i,startB+j);
			}
		}
		//compute trsv kernel for each row in block
		for(std::size_t j = 0; j != curBlockSize2; ++j){
			for (std::size_t n = 1; n <= size; ++n) {
				std::size_t i = size-n;
				value_typeB bi = block[j][i];
				if(!Unit){
					block[j][i] *= A()(i,i);
				}
				for(std::size_t k=i+1; k != size; ++k){
					block[j][k] += bi * A()(k,i);
				}
			}
		}
		//copy block back
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t j = 0; j != curBlockSize2; ++j){
				B()(i,startB+j) = block[j][i];
			}
		}
	}
}

//Lower triangular - matrix(column-major)
template<std::size_t maxBlockSize1,std::size_t maxBlockSize2, bool Unit, class MatA, class MatB>
void trmm_block(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag> &B,
        boost::mpl::false_, //Lower
	column_major // B is column-major
){
	typedef typename MatB::value_type value_typeB;

	std::size_t size = A().size1();
		
	//compute trsv kernel for each column
	for(std::size_t j = 0; j != B().size2(); ++j){
		for (std::size_t n = 1; n <= size; ++n) {
			std::size_t i = size-n;
			value_typeB bi = B()(i,j);
			if(!Unit){
				B()(i,j) *= A()(i,i);
			}
			for(std::size_t k=i+1; k != size; ++k){
				B()(k,j) += bi * A()(k,i);
			}
		}
	}
}


//Upper triangular - matrix(row-major)
template<std::size_t maxBlockSize1,std::size_t maxBlockSize2, bool Unit, class MatA, class MatB>
void trmm_block(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag> &B,
        boost::mpl::true_, //Upper
	row_major // B is row-major
){
	SIZE_CHECK(A().size1() <= maxBlockSize1);
	typedef typename MatB::value_type value_typeB;

	value_typeB block[maxBlockSize2][maxBlockSize1];
	std::size_t numBlocks = (B().size2()+maxBlockSize2-1)/maxBlockSize2;
	std::size_t size = A().size1();
	for(std::size_t i = 0; i != numBlocks; ++i){
		std::size_t startB= i*maxBlockSize2;
		std::size_t curBlockSize2 =std::min(maxBlockSize2, B().size2() - startB);
		
		//copy block transposed in memory
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t j = 0; j != curBlockSize2; ++j){
				block[j][i] = B()(i,startB+j);
			}
		}
		//compute trsv kernel for each column in block
		for(std::size_t j = 0; j != curBlockSize2; ++j){
			for (std::size_t i = 0; i < size; ++ i) {
				if(!Unit){
					block[j][i]  *= A()(i,i);
				}
				for(std::size_t k=i+1; k != size; ++k){
					block[j][i]  += A()(i,k) * block[j][k];
				}
			}
		}
		//copy block back
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t j = 0; j != curBlockSize2; ++j){
				B()(i,startB+j) = block[j][i];
			}
		}
	}
}

//Upper triangular - matrix(column-major)
template<std::size_t maxBlockSize1,std::size_t maxBlockSize2, bool Unit, class MatA, class MatB>
void trmm_block(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag> &B,
        boost::mpl::true_, //Upper
	column_major // B is column-major
){
	std::size_t size = A().size1();
		
	//compute trsv kernel for each column
	for(std::size_t j = 0; j != B().size2(); ++j){
		for (std::size_t i = 0; i < size; ++ i) {
			if(!Unit){
				B()(i,j) *= A()(i,i);
			}
			for(std::size_t k=i+1; k != size; ++k){
				B()(i,j) +=A()(i,k) * B()(k,j) ;
			}
		}
	}
}

template <bool Upper,bool Unit,typename MatA, typename MatB>
void trmm_recursive(
	matrix_expression<MatA, cpu_tag> const& Afull, 
	matrix_expression<MatB, cpu_tag> & BFull,
	std::size_t start,
	std::size_t end
){
	auto A = subrange(Afull,start,end,start,end);
	auto B = rows(BFull,start,end);
	std::size_t size = A.size1();
	std::size_t split = A.size1()/2;
	auto BFront = rows(B,0,split);
	auto Bback = rows(B,split,size);
	//if the matrix is small enough call the computation kernel directly for the block
	if(A.size1() < 32){
		trmm_block<32,16,Unit>(A,B,boost::mpl::bool_<Upper>(), typename MatB::orientation());
	}
	//otherwise run the kernel recursively
	else if(Upper){ //Upper triangular case
		trmm_recursive<Upper,Unit>(Afull, BFull,start,start+split);
		kernels::gemm(subrange(A,0,split,split,size), Bback, BFront, 1.0);
		trmm_recursive<Upper,Unit>(Afull, BFull,start+split,end);
	}else{// Lower triangular caste
		trmm_recursive<Upper,Unit>(Afull, BFull,start+split,end);
		kernels::gemm(subrange(A,split,size,0,split), BFront, Bback, 1.0);
		trmm_recursive<Upper,Unit>(Afull, BFull,start,start+split);
	}
}
//main kernel runs the kernel above recursively and calls gemv
template <bool Upper,bool Unit,typename MatA, typename MatB>
void trmm(
	matrix_expression<MatA, cpu_tag> const& A, 
	matrix_expression<MatB, cpu_tag>& B,
	boost::mpl::false_ //unoptimized
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == B().size1());
	
	trmm_recursive<Upper,Unit>(A,B,0,A().size1());
	
	
}

}}}

#endif
