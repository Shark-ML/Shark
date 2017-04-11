/*!
 *
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

#ifndef REMORA_KERNELS_DEFAULT_GEMM_HPP
#define REMORA_KERNELS_DEFAULT_GEMM_HPP

#include "../gemv.hpp"//for dispatching to gemv
#include "../../assignment.hpp"//plus_assign
#include "../../vector.hpp"//sparse gemm needs temporary vector
#include "../../detail/matrix_proxy_classes.hpp"//matrix row,column,transpose,range
#include <type_traits> //std::false_type marker for unoptimized, std::common_type

namespace remora{namespace bindings {


// Dense-Sparse gemm
template <class E1, class E2, class M, class Orientation>
void gemm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, row_major, Orientation,
	dense_tag, sparse_tag
){
	for (std::size_t i = 0; i != e1().size1(); ++i) {
		matrix_row<M> row_m(m(),i);
		matrix_row<typename const_expression<E1>::type> row_e1(e1(),i);
		matrix_transpose<typename const_expression<E2>::type> trans_e2(e2());
		kernels::gemv(trans_e2,row_e1,row_m,alpha);
	}
}

template <class E1, class E2, class M>
void gemm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, column_major, column_major,
	dense_tag, sparse_tag
){
	typedef matrix_transpose<M> Trans_M;
	typedef matrix_transpose<typename const_expression<E2>::type> Trans_E2;
	Trans_M trans_m(m());
	Trans_E2 trans_e2(e2());
	for (std::size_t j = 0; j != e2().size2(); ++j) {
		matrix_row<Trans_M> column_m(trans_m,j);
		matrix_row<Trans_E2> column_e2(trans_e2,j);
		kernels::gemv(e1,column_e2,column_m,alpha);
	}
}

template <class E1, class E2, class M>
void gemm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, column_major, row_major,
	dense_tag, sparse_tag
){
	for (std::size_t k = 0; k != e1().size2(); ++k) {
		for(std::size_t i = 0; i != e1().size1(); ++i){
			matrix_row<M> row_m(m(),i);
			matrix_row<typename const_expression<E2>::type> row_e2(e2(),k);
			plus_assign(row_m,row_e2,alpha * e1()(i,k));
		}
	}
}

// Sparse-Dense gemm
template <class E1, class E2, class M, class Orientation>
void gemm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, row_major, Orientation,
	sparse_tag, dense_tag
){
	for (std::size_t i = 0; i != e1().size1(); ++i) {
		matrix_row<M> row_m(m(),i);
		matrix_row<E1> row_e1(e1(),i);
		matrix_transpose<E2> trans_e2(e2());
		kernels::gemv(trans_e2,row_e1,row_m,alpha);
	}
}

template <class E1, class E2, class M>
void gemm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, column_major, column_major,
	sparse_tag, dense_tag
){
	typedef matrix_transpose<M> Trans_M;
	typedef matrix_transpose<typename const_expression<E2>::type> Trans_E2;
	Trans_M trans_m(m());
	Trans_E2 trans_e2(e2());
	for (std::size_t j = 0; j != e2().size2(); ++j) {
		matrix_row<Trans_M> column_m(trans_m,j);
		matrix_row<Trans_E2> column_e2(trans_e2,j);
		kernels::gemv(e1,column_e2,column_m,alpha);
	}
}

template <class E1, class E2, class M>
void gemm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, column_major, row_major,
	sparse_tag, dense_tag
){
	for (std::size_t k = 0; k != e1().size2(); ++k) {
		auto e1end = e1().column_end(k);
		for(auto e1pos = e1().column_begin(k); e1pos != e1end; ++e1pos){
			std::size_t i = e1pos.index();
			matrix_row<M> row_m(m(),i);
			matrix_row<typename const_expression<E2>::type> row_e2(e2(),k);
			plus_assign(row_m,row_e2,alpha * (*e1pos));
		}
	}
}

// Sparse-Sparse gemm
template<class M, class E1, class E2>
void gemm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, row_major, row_major,
	sparse_tag, sparse_tag
) {
	typedef typename M::value_type value_type;
	value_type zero = value_type();
	vector<value_type> temporary(e2().size2(), zero);//dense vector for quick random access
	matrix_transpose<typename const_expression<E2>::type> e2trans(e2());
	for (std::size_t i = 0; i != e1().size1(); ++i) {
		matrix_row<typename const_expression<E1>::type> rowe1(e1(),i);
		kernels::gemv(e2trans,rowe1,temporary,alpha);
		auto insert_pos = m().row_begin(i);
		for (std::size_t j = 0; j != temporary.size(); ++ j) {
			if (temporary(j) != zero) {
				//find element with that index
				auto row_end = m().row_end(i);
				while(insert_pos != row_end && insert_pos.index() < j)
					++insert_pos;
				//check if element exists
				if(insert_pos != row_end && insert_pos.index() == j){
					*insert_pos += temporary(j);
				}else{//create new element
					insert_pos = m().set_element(insert_pos,j,temporary(j));
				}
				//~ m()(i,j) += temporary(j);
				temporary(j) = zero; // delete element
			}
		}
	}
}

template<class M, class E1, class E2>
void gemm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, row_major, column_major,
	sparse_tag, sparse_tag
) {
	typedef matrix_transpose<M> Trans_M;
	typedef matrix_transpose<typename const_expression<E2>::type> Trans_E2;
	Trans_M trans_m(m());
	Trans_E2 trans_e2(e2());
	for (std::size_t j = 0; j != e2().size2(); ++j) {
		matrix_row<Trans_M> column_m(trans_m,j);
		matrix_row<Trans_E2> column_e2(trans_e2,j);
		kernels::gemv(e1,column_e2,column_m,alpha);
	}
}

template <class E1, class E2, class M, class Orientation>
void gemm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha,
	row_major, column_major, Orientation o,
	sparse_tag t1, sparse_tag t2
){
	//best way to compute this is to transpose e1 in memory. alternative would be
	// to compute outer products, which is a no-no.
	typename transposed_matrix_temporary<E1>::type e1_trans(e1);
	gemm(e1_trans,e2,m,alpha,row_major(),row_major(),o,t1,t2);
}

}}

#endif
