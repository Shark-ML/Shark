/*!
 * \brief      Permutations of vectors and matrices
 * 
 * \author      O. Krause
 * \date        2013
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
#ifndef REMORA_PERMUTATION_HPP
#define REMORA_PERMUTATION_HPP

#include "dense.hpp"

namespace remora {
struct permutation_matrix:public vector<int> {
	// Construction and destruction
	explicit permutation_matrix(size_type size):vector<int> (size){
		for (int i = 0; i < (int)size; ++ i)
			(*this)(i) = i;
	}

	// Assignment
	permutation_matrix &operator = (permutation_matrix const& m) {
		vector<int>::operator = (m);
		return *this;
	}
};

///\brief implements row pivoting at matrix A using permutation P
///
///by convention it is not allowed that P()(i) < i. 
template<class VecP, class M>
void swap_rows(vector_expression<VecP, cpu_tag> const& P, matrix_expression<M, cpu_tag>& A){
	for (std::size_t i = 0; i != P().size(); ++ i)
		A().swap_rows(i,P()(i));
}

///\brief implements column pivoting of vector A using permutation P
///
///by convention it is not allowed that P()(i) < i. 
template<class VecP, class V>
void swap_rows(vector_expression<VecP, cpu_tag> const& P, vector_expression<V, cpu_tag>& v){
	for (std::size_t i = 0; i != P().size(); ++ i)
		std::swap(v()(i),v()(P()(i)));
}

///\brief implements the inverse row pivoting of vector v using permutation P
///
///This is the inverse operation to swap_rows. 
template<class VecP, class V>
void swap_rows_inverted(vector_expression<VecP, cpu_tag> const& P, vector_expression<V, cpu_tag>& v){
	for(std::size_t i = P().size(); i != 0; --i){
		std::size_t k = i-1;
		if(k != std::size_t(P()(k))){
			using std::swap;
			swap(v()(k),v()(P()(k)));
		}
	}
}

///\brief implements column pivoting at matrix A using permutation P
///
///by convention it is not allowed that P(i) < i. 
template<class VecP, class M>
void swap_columns(vector_expression<VecP, cpu_tag> const& P, matrix_expression<M, cpu_tag>& A){
	for(std::size_t i = 0; i != P().size(); ++i)
		A().swap_columns(i,P()(i));
}

///\brief implements the inverse row pivoting at matrix A using permutation P
///
///This is the inverse operation to swap_rows. 
template<class VecP, class M>
void swap_rows_inverted(vector_expression<VecP, cpu_tag> const& P, matrix_expression<M, cpu_tag>& A){
	for(std::size_t i = P().size(); i != 0; --i){
		A().swap_rows(i-1,P()(i-1));
	}
}

///\brief implements the inverse column pivoting at matrix A using permutation P
///
///This is the inverse operation to swap_columns. 
template<class VecP, class M>
void swap_columns_inverted(vector_expression<VecP, cpu_tag> const& P, matrix_expression<M, cpu_tag>& A){
	for(std::size_t i = P().size(); i != 0; --i){
		A().swap_columns(i-1,P()(i-1));
	}
}

///\brief Implements full pivoting at matrix A using permutation P
///
///full pivoting does swap rows and columns such that the diagonal element
///A_ii is then at position A_P(i)P(i)
///by convention it is not allowed that P(i) < i. 
template<class VecP, class M>
void swap_full(vector_expression<VecP, cpu_tag> const& P, matrix_expression<M, cpu_tag>& A){
	for(std::size_t i = 0; i != P().size(); ++i){
		A().swap_rows(i,P()(i));
		A().swap_columns(i,P()(i));
	}
}
///\brief implements the inverse full pivoting at matrix A using permutation P
///
///This is the inverse operation to swap_full. 
template<class VecP, class M>
void swap_full_inverted(vector_expression<VecP, cpu_tag> const& P, matrix_expression<M, cpu_tag>& A){
	for(std::size_t i = P().size(); i != 0; --i){
		A().swap_rows(i-1,P()(i-1));
		A().swap_columns(i-1,P()(i-1));
	}
}

}
#endif
