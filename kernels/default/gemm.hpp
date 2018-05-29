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
#include "../vector_assign.hpp" //assignment of vectors
#include "../../dense.hpp"//sparse gemm needs temporary vector
#include "../../proxy_expressions.hpp"//matrix row,column,transpose,range
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
		auto row_m = row(m,i);
		kernels::gemv(trans(e2),row(e1,i),row_m,alpha);
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
	for (std::size_t j = 0; j != e2().size2(); ++j) {
		auto column_m = column(m,j);
		kernels::gemv(e1,column(e2,j),column_m,alpha);
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
	typedef typename M::value_type value_type;
	typedef device_traits<cpu_tag>::multiply_and_add<value_type> MultAdd;
	for (std::size_t k = 0; k != e1().size2(); ++k) {
		for(std::size_t i = 0; i != e1().size1(); ++i){
			auto row_m = row(m,i);
			kernels::assign(row_m, row(e2,k), MultAdd(alpha * e1()(i,k)));
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
		auto row_m = row(m,i);
		kernels::gemv(trans(e2),row(e1,i),row_m,alpha);
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
	for (std::size_t j = 0; j != e2().size2(); ++j) {
		auto column_m = column(m,j);
		kernels::gemv(e1,column(e2,j),column_m,alpha);
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
	typedef typename M::value_type value_type;
	typedef device_traits<cpu_tag>::multiply_and_add<value_type> MultAdd;
	for (std::size_t k = 0; k != e1().size2(); ++k) {
		auto e1end = e1().major_end(k);
		for(auto e1pos = e1().major_begin(k); e1pos != e1end; ++e1pos){
			std::size_t i = e1pos.index();
			auto row_m = row(m,i);
			kernels::assign(row_m, row(e2,k), MultAdd(alpha * (*e1pos)));
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
	for (std::size_t i = 0; i != e1().size1(); ++i) {
		kernels::gemv(trans(e2),row(e1,i),temporary,alpha);
		auto insert_pos = m().major_begin(i);
		for (std::size_t j = 0; j != temporary.size(); ++ j) {
			if (temporary(j) != zero) {
				//find element with that index
				auto row_end = m().major_end(i);
				while(insert_pos != row_end && insert_pos.index() < j)
					++insert_pos;
				//check if element exists
				if(insert_pos != row_end && insert_pos.index() == j){
					*insert_pos += temporary(j);
				}else{//create new element
					insert_pos = m().set_element(insert_pos,j,temporary(j));
				}
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
	for (std::size_t j = 0; j != e2().size2(); ++j) {
		auto column_m = column(m,j);
		kernels::gemv(e1,column(e2,j),column_m,alpha);
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
