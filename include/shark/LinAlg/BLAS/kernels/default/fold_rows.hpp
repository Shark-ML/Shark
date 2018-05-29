/*!
 * 
 *
 * \brief       Folds the rows of a row-major or column major matrix.
 *
 * \author      O. Krause
 * \date        2018
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

#ifndef REMORA_KERNELS_DEFAULT_FOLD_ROWS_HPP
#define REMORA_KERNELS_DEFAULT_FOLD_ROWS_HPP

#include "../../expression_types.hpp"//for vector/matrix_expression
#include "../../detail/traits.hpp"

namespace remora{namespace bindings{
	
template<class F, class G, class M,class V>
void fold_rows(
	matrix_expression<M, cpu_tag> const& A, 
	vector_expression<V, cpu_tag>& v,
	F f,
	G g,
	row_major
){
	for(std::size_t i = 0; i != v().size(); ++i){
		auto end = A().major_end(i);
		auto pos = A().major_begin(i);
		typename V::value_type s = *pos;
		++pos;
		for(; pos != end; ++pos){
			s = f(s,*pos);
		}
		v()(i) += g(s);
	}
}

template<class F, class G, class M,class V>
void fold_rows(
	matrix_expression<M, cpu_tag> const& A, 
	vector_expression<V, cpu_tag>& v,
	F f,
	G g,
	column_major
){
	std::size_t n = v().size();
	const std::size_t BLOCK_SIZE = 16;
	typename V::value_type storage[BLOCK_SIZE];
	std::size_t numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; 
	
	for(std::size_t b = 0; b != numBlocks; ++b){
		std::size_t start = b * BLOCK_SIZE;
		std::size_t cur_size = std::min(BLOCK_SIZE, n - start);
		for(std::size_t i = 0; i != cur_size; ++i){
			storage[i] = A()(start + i, 0);
		}
		for(std::size_t j = 1; j != A().size2(); ++j){
			for(std::size_t i = 0; i != cur_size; ++i){
				storage[i] = f(storage[i], A()(start + i, j));
			}
		}
		for(std::size_t i = 0; i != cur_size; ++i){
			v()(start + i) += g(storage[i]);
		}
	}
}

//dispatcher for triangular matrix
template<class F, class G, class M,class V,class Orientation,class Triangular>
void fold_rows(
	matrix_expression<M, cpu_tag> const& A, 
	vector_expression<V, cpu_tag>& v,
	F f,
	G g,
	triangular<Orientation,Triangular>
){
	fold_rows(A,v, f, g, Orientation());
}

}}

#endif
