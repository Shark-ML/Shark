//===========================================================================
/*!
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
//===========================================================================
#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_TRMV_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_TRMV_HPP

#include <boost/mpl/bool.hpp>
#include "../gemv.hpp"

namespace shark{ namespace blas{ namespace bindings{

// first block-kernels which compute the small triangular parts

//Lower triangular - vector
template<bool Unit, class MatA, class V>
void trmv_block(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag> &b,
        boost::mpl::false_ //Lower
){
	typedef typename V::value_type value_typeV;
	std::size_t size = A().size1();
	for (std::size_t n = 1; n <= size; ++n) {
		std::size_t i = size-n;
		value_typeV bi = b()(i);
		if(!Unit){
			b()(i) *= A()(i,i);
		}
		noalias(subrange(b,i+1,size))+= bi * subrange(column(A,i),i+1,size);
	}
}

//upper triangular(row-major)-vector
template<bool Unit, class MatA, class V>
void trmv_block(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag>& b,
        boost::mpl::true_ //Upper
){
	std::size_t size = A().size1();
	for (std::size_t i = 0; i < size; ++ i) {
		if(!Unit){
			b()(i) *= A()(i,i);
		}
		b()(i) += inner_prod(subrange(row(A,i),i+1,size),subrange(b,i+1,size));
	}
}

// recursive block-based kernel
template<bool Upper, bool Unit, class MatA, class V>
void trmv_recursive(
	matrix_expression<MatA, cpu_tag> const& AFull,
	vector_expression<V, cpu_tag>& bFull,
	std::size_t start,
	std::size_t end
){
	std::size_t const blockSize = 32;
	auto A = subrange(AFull,start,end,start,end);
	auto b = subrange(bFull,start,end);
	std::size_t size = A.size1();
	std::size_t split = A.size1()/2;
	auto bfront = subrange(b,0,split);
	auto bback = subrange(b,split,size);
	//if the matrix is small enough call the computation kernel directly for the block
	if(A.size1() < blockSize){
		trmv_block<Unit>(A,b,boost::mpl::bool_<Upper>());
	}
	//otherwise run the kernel recursively
	else if(Upper){ //Upper triangular case
		trmv_recursive<Upper,Unit>(AFull, bFull,start,start+split);
		kernels::gemv(subrange(A,0,split,split,size), bback, bfront, 1.0 );
		trmv_recursive<Upper,Unit>(AFull, bFull,start+split,end);
	}else{// Lower triangular caste
		trmv_recursive<Upper,Unit>(AFull, bFull,start+split,end);
		kernels::gemv(subrange(A,split,size,0,split), bfront, bback, 1.0);
		trmv_recursive<Upper,Unit>(AFull, bFull,start,start+split);
	}
}

//main kernel runs the kernel above recursively and calls gemv
template <bool Upper,bool Unit,typename MatA, typename V>
void trmv(
	matrix_expression<MatA, cpu_tag> const& A, 
	vector_expression<V, cpu_tag> & b,
	boost::mpl::false_ //unoptimized
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == b().size());
	
	trmv_recursive<Upper,Unit>(A,b,0,A().size1());
}

}}}
#endif
