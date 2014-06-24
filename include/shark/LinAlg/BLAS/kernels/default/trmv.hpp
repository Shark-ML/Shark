//===========================================================================
/*!
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2010
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
//===========================================================================
#ifndef SHARK_LINALG_BLAS_UBLAS_KERNELS_DEFAULT_TRMV_HPP
#define SHARK_LINALG_BLAS_UBLAS_KERNELS_DEFAULT_TRMV_HPP

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
	//TODO: REASONABLY FAST IMPLEMENTATION
	typedef typename TriangularA::value_type value_type;
	
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
	typedef typename TriangularA::value_type value_type;
	
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
	//TODO: REASONABLY FAST IMPLEMENTATION
	std::size_t size = A().size1();
	for (std::size_t i = 0; i < size; ++ i) {
		if(!Unit){
			b()(i) *= A()(i,i);
		}
		b()(i) += inner_prod(subrange(row(A,i),i+1,size),subrange(b,i+1,size));
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
