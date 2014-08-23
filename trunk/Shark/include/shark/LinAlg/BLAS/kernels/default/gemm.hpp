/*!
 * 
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

#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_GEMM_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_GEMM_HPP

#include "../gemv.hpp"
#include "../../matrix_proxy.hpp"
#include "../../vector.hpp"
#include <boost/mpl/bool.hpp>

namespace shark { namespace blas { namespace bindings {

//we dispatch gemm: A=B*C in the following like that:
//all 	orientations of A,B,C
//iterator category of B and C. We assume A to have a meaningful storage category
	
// basic dispatching towards the kernels works in categories. 
// We explain it here because the implementation needs to be inverted as reducing one case to another
// requires that the resulting case has already been implemented. Thus the most general cases are at the end of the file.
//
// 1. we dispatch for the orientation of the result using the relation A=B*C <=> A^T = C^T B^T
// thus we can assume for all compute kernels that the first argument is row_major. 
// 2. if B is row_major as well we can cast the computation in terms of matrix-vector products
// computing A row by row (note that we use a specialised kernel for all-sparse)
// 3. If B is column_major we can dispatch as in the following:
// 3.1 if B is sparse, transpose B in memory. This is a bit of memory overhead but is often fast (and easy)
// 3.2 else cast the computation in terms of an outer product if the C is row_major
// 3.3 for B and C column major there are specialised kernels for every combination
	
	
//general case: result and first argument row_major (2.)
//=> compute as a sequence of matrix-vector products over the rows of the first argument
template<class M, class E1, class E2, class Orientation2,class Tag1,class Tag2>
void gemm_impl(
	matrix_expression<E1> const& e1,
	matrix_expression<E2> const& e2,
	matrix_expression<M>& m,
	typename M::value_type alpha,
	row_major, row_major, Orientation2, 
	Tag1, Tag2
) {
	for (std::size_t i = 0; i != e1().size1(); ++i) {
		matrix_row<M> mat_row(m(),i);
		kernels::gemv(trans(e2),row(e1,i),mat_row,alpha);
	}
}

//case: sparse column_major first argument (3.1)
//=> transpose in memory
template<class M, class E1, class E2, class Orientation, class Tag>
void gemm_impl(
	matrix_expression<E1> const& e1,
	matrix_expression<E2> const& e2,
	matrix_expression<M>& m,
	typename M::value_type alpha,
	row_major, column_major, Orientation o,
	sparse_bidirectional_iterator_tag t1, Tag t2
) {
	typename transposed_matrix_temporary<E1>::type e1_trans(e1);
	gemm_impl(e1_trans,e2,m,alpha,row_major(),row_major(),o,t1,t2);
}

//case: result and second argument row_major, first argument dense column major (3.2)
//=> compute as a sequence of outer products. 
// Note that this is likely to be slow if E2 is sparse and the result is also sparse. However choosing 
// M as sparse is stupid in most cases.
template<class M, class E1, class E2,class Tag>
void gemm_impl(
	matrix_expression<E1> const& e1,
	matrix_expression<E2> const& e2,
	matrix_expression<M>& m,
	typename M::value_type alpha,
	row_major, column_major, row_major,
	dense_random_access_iterator_tag, Tag
) {
	for (std::size_t j = 0; j != e1().size2(); ++j) {
		noalias(m) += alpha * outer_prod(column(e1,j),row(e2,j));
	}
}

//special case of all row-major for sparse matrices
template<class M, class E1, class E2>
void gemm_impl(
	matrix_expression<E1> const& e1,
	matrix_expression<E2> const& e2,
	matrix_expression<M>& m,
	typename M::value_type alpha,
	row_major, row_major, row_major, 
	sparse_bidirectional_iterator_tag, sparse_bidirectional_iterator_tag
) {
	typedef typename M::value_type value_type;
	value_type zero = value_type();
	vector<value_type> temporary(e2().size2(), zero);
	for (std::size_t i = 0; i != e1().size1(); ++i) {
		kernels::gemv(trans(e2),row(e1,i),temporary,alpha);
		for (std::size_t j = 0; j != temporary.size(); ++ j) {
			if (temporary(j) != zero) {
				m()(i, j) += temporary(j);//fixme: better use something like insert
				temporary(j) = zero;
			}
		}
	}
}

	


// case 3.3
//now we only need to handle the case that E1 and E2 are column major and M row_major. This
// is a special case for all matrix types (except sparse column_major E1)

//dense-sparse
template<class M, class E1, class E2>
void gemm_impl(
	matrix_expression<E1> const& e1,
	matrix_expression<E2> const& e2,
	matrix_expression<M>& m,
	typename M::value_type alpha,
	row_major, column_major, column_major,
	dense_random_access_iterator_tag, sparse_bidirectional_iterator_tag
) {
	//compute the product row-wise
	for (std::size_t i = 0; i != m().size1(); ++i) {
		matrix_row<M> mat_row(m(),i);
		kernels::gemv(trans(e2),row(e1,i),mat_row,alpha);
	}
}

//dense-dense
template<class M, class E1, class E2>
void gemm_impl(
	matrix_expression<E1> const& e1,
	matrix_expression<E2> const& e2,
	matrix_expression<M>& m,
	typename M::value_type alpha,
	row_major r, column_major, column_major, 
	dense_random_access_iterator_tag t, dense_random_access_iterator_tag
) {
	//compute blockwise and write the transposed block.
	std::size_t blockSize = 16;
	typedef typename M::value_type value_type;
	typedef typename matrix_temporary<M>::type BlockStorage;
	BlockStorage blockStorage(blockSize,blockSize);
	
	typedef typename M::size_type size_type;
	size_type size1 = m().size1();
	size_type size2 = m().size2();
	for (size_type i = 0; i < size1; i+= blockSize){
		for (size_type j = 0; j < size2; j+= blockSize){
			std::size_t blockSizei = std::min(blockSize,size1-i);
			std::size_t blockSizej = std::min(blockSize,size2-j);
			matrix_range<matrix<value_type> > transBlock=subrange(blockStorage,0,blockSizej,0,blockSizei);
			transBlock.clear();
			//reduce to all row-major case by using
			//A_ij=B^iC_j <=> A_ij^T = (C_j)^T (B^i)^T  
			gemm_impl(
				trans(columns(e2,j,j+blockSizej)),
				trans(rows(e1,i,i+blockSizei)),
				transBlock,alpha,
				r,r,r,//all row-major
				t,t //both targets are dense
			);
			//write transposed block to the matrix
			noalias(subrange(m,i,i+blockSizei,j,j+blockSizej))+=trans(transBlock);
		}
	}
}

//general case: column major result case (1.0)
//=> transformed to row_major using A=B*C <=> A^T = C^T B^T
template<class M, class E1, class E2, class Orientation1, class Orientation2, class Tag1, class Tag2>
void gemm_impl(
	matrix_expression<E1> const& e1,
	matrix_expression<E2> const& e2,
	matrix_expression<M>& m,
	typename M::value_type alpha,
	column_major, Orientation1, Orientation2, 
	Tag1, Tag2
){
	matrix_transpose<M> transposedM(m());
	typedef typename Orientation1::transposed_orientation transpO1;
	typedef typename Orientation2::transposed_orientation transpO2;
	gemm_impl(trans(e2),trans(e1),transposedM,alpha,row_major(),transpO2(),transpO1(), Tag2(),Tag1());
}

//dispatcher
template<class M, class E1, class E2>
void gemm(
	matrix_expression<E1> const& e1,
	matrix_expression<E2> const& e2,
	matrix_expression<M>& m,
	typename M::value_type alpha,
	boost::mpl::false_
) {
	SIZE_CHECK(m().size1() == e1().size1());
	SIZE_CHECK(m().size2() == e2().size2());
	
	typedef typename M::orientation ResultOrientation;
	typedef typename E1::orientation E1Orientation;
	typedef typename E2::orientation E2Orientation;
	typedef typename major_iterator<E1>::type::iterator_category E1Category;
	typedef typename major_iterator<E2>::type::iterator_category E2Category;
	
	gemm_impl(e1, e2, m,alpha,
		ResultOrientation(),E1Orientation(),E2Orientation(),
		E1Category(),E2Category()
	);
}

}}}

#endif
