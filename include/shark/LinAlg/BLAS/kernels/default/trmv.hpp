//===========================================================================
/*!
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2010
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
//===========================================================================
#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_TRMV_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_TRMV_HPP

#include "../../matrix_proxy.hpp"
#include "../../vector_expression.hpp"
#include <boost/mpl/bool.hpp>

namespace shark{ namespace blas{ namespace bindings{
	
//Lower triangular(row-major) - vector
template<bool Unit, class TriangularA, class V>
void trmv_impl(
	matrix_expression<TriangularA> const& A,
	vector_expression<V> &b,
        boost::mpl::false_, row_major
){
	typedef typename TriangularA::value_type value_typeA;
	typedef typename V::value_type value_typeV;
	std::size_t size = A().size1();
	std::size_t const blockSize = 128;
	std::size_t numBlocks = size/blockSize;
	if(numBlocks*blockSize < size) ++numBlocks; 
	
	//this implementation partitions A into
	//a set of panels, where a Panel is a set
	// of columns. We start with the last panel
	//and compute the product of it with the part of the vector
	// and than just add the previous panel on top etc.
	
	//tmporary storage for subblocks of b
	value_typeV valueStorage[blockSize];
	
	for(std::size_t bi = 1; bi <= numBlocks; ++bi){
		std::size_t startbi = blockSize*(numBlocks-bi);
		std::size_t sizebi = std::min(blockSize,size-startbi);
		dense_vector_adaptor<value_typeA> values(valueStorage,sizebi);
		
		//store and save the values of b we ar enow changing
		noalias(values) = subrange(b,startbi,startbi+sizebi);
		
		//multiply with triangular element
		for (std::size_t i = 0; i != sizebi; ++i) {
			std::size_t posi = startbi+i;
			b()(posi) = 0;
			for(std::size_t j = 0; j < i; ++j){
				b()(posi) += A()(posi,startbi+j)*values(j);
			}
			b()(posi) += values(i)*(Unit? value_typeA(1):A()(posi,posi));
		}
		//now compute the remaining inner products
		for(std::size_t posi = startbi+sizebi; posi != size; ++posi){
			b()(posi) += inner_prod(values,subrange(row(A,posi),startbi,startbi+sizebi));
		}
	}
}

//upper triangular(row-major)-vector
template<bool Unit, class TriangularA, class V>
void trmv_impl(
	matrix_expression<TriangularA> const& A,
	vector_expression<V>& b,
        boost::mpl::true_, row_major
){
	std::size_t size = A().size1();
	for (std::size_t i = 0; i < size; ++ i) {
		if(!Unit){
			b()(i) *= A()(i,i);
		}
		b()(i) += inner_prod(subrange(row(A,i),i+1,size),subrange(b,i+1,size));
	}
}

//Lower triangular(column-major) - vector
template<bool Unit, class TriangularA, class V>
void trmv_impl(
	matrix_expression<TriangularA> const& A,
	vector_expression<V> &b,
        boost::mpl::false_, column_major
){
	
	std::size_t size = A().size1();
	for (std::size_t n = 1; n <= size; ++n) {
		std::size_t i = size-n;
		double bi = b()(i);
		if(!Unit){
			b()(i) *= A()(i,i);
		}
		noalias(subrange(b,i+1,size))+= bi * subrange(column(A,i),i+1,size);
	}
}

//upper triangular(column-major)-vector
template<bool Unit, class TriangularA, class V>
void trmv_impl(
	matrix_expression<TriangularA> const& A,
	vector_expression<V>& b,
        boost::mpl::true_, column_major
){
	typedef typename TriangularA::value_type value_typeA;
	typedef typename V::value_type value_typeV;
	std::size_t size = A().size1();
	std::size_t const blockSize = 128;
	std::size_t numBlocks = size/blockSize;
	if(numBlocks*blockSize < size) ++numBlocks; 
	
	//this implementation partitions A into
	//a set of panels, where a Panel is a set
	// of rows. We start with the first panel
	//and compute the product of it with the part of the vector
	// and than just add the next panel on top etc.
	
	//tmporary storage for subblocks of b
	value_typeV valueStorage[blockSize];
	
	for(std::size_t bj = 0; bj != numBlocks; ++bj){
		std::size_t startbj = blockSize*bj;
		std::size_t sizebj = std::min(blockSize,size-startbj);
		dense_vector_adaptor<value_typeA> values(valueStorage,sizebj);
		
		//store and save the values of b we ar enow changing
		noalias(values) = subrange(b,startbj,startbj+sizebj);
		subrange(b,startbj,startbj+sizebj).clear();
		//multiply with triangular element
		for (std::size_t j = 0; j != sizebj; ++j) {
			std::size_t posj = startbj+j;
			for(std::size_t i = 0; i < j; ++i){
				b()(startbj+i) += A()(startbj+i,posj)*values(j);
			}
			b()(posj) += values(j)*(Unit? 1.0:A()(posj,posj));
		}
		//now compute the remaining inner products
		for(std::size_t posj = startbj+sizebj; posj != size; ++posj){
			noalias(subrange(b,startbj,startbj+sizebj)) += b()(posj)*subrange(column(A,posj),startbj,startbj+sizebj);
		}
	}
}

//dispatcher
template <bool Upper,bool Unit,typename TriangularA, typename V>
void trmv(
	matrix_expression<TriangularA> const& A, 
	vector_expression<V> & b,
	boost::mpl::false_//unoptimized
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == b().size());
	trmv_impl<Unit>(A, b, boost::mpl::bool_<Upper>(), typename TriangularA::orientation());
}

}}}
#endif
