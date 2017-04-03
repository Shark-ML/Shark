//===========================================================================
/*!
 * 
 *
 * \brief       Storage Types of matrix expressions
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
//===========================================================================

#ifndef REMORA_DETAIL_STORAGE_HPP
#define REMORA_DETAIL_STORAGE_HPP

#include "structure.hpp"

namespace remora{
	
struct unknown_storage{
	typedef unknown_tag storage_tag;
	typedef unknown_storage row_storage;
};

template<class T>
struct dense_vector_storage{
	typedef dense_tag storage_tag;
	T* values;
	std::size_t stride;
	
	dense_vector_storage<T> sub_region(std::size_t offset){
		return {values+offset*stride, stride};
	}
};

template<class T, class I>
struct sparse_vector_storage{
	typedef sparse_tag storage_tag;
	T* values;
	I* indices;
	std::size_t nnz;
};

template<class T>
struct dense_matrix_storage{
	typedef dense_tag storage_tag;
	typedef dense_vector_storage<T> row_storage;
	T* values;
	std::size_t leading_dimension;
	
	template<class Orientation>
	dense_matrix_storage<T> sub_region(std::size_t offset1, std::size_t offset2, Orientation){
		std::size_t offset_major = Orientation::index_M(offset1,offset2);
		std::size_t offset_minor = Orientation::index_m(offset1,offset2);
		return {values+offset_major*leading_dimension+offset_minor, leading_dimension};
	}
	
	template<class Orientation>
	row_storage row(std::size_t i, Orientation){
		return {values + i * Orientation::index_M(leading_dimension,(std::size_t)1), Orientation::index_m(leading_dimension,(std::size_t)1)};
	}
	template<class Orientation>
	row_storage diag(){
		return {values, leading_dimension+1};
	}
};

template<class T>
struct packed_matrix_storage{
	typedef packed_tag storage_tag;
	typedef dense_vector_storage<T> row_storage;
	T* values;
	std::size_t nnz;
};

template<class T, class I>
struct sparse_matrix_storage{
	typedef sparse_tag storage_tag;
	typedef sparse_vector_storage<T,I> row_storage;
	T* values;
	I* indices;
	I* outer_indices_begin;
	I* outer_indices_end;
	
	template<class Orientation>
	row_storage row(std::size_t i, Orientation){
		static_assert(std::is_same<Orientation,row_major>::value, "sparse matrix has wrong orientation for row/column");
		return {values + outer_indices_begin[i], indices + outer_indices_begin[i],outer_indices_end[i] - outer_indices_begin[i]};
	}
};
}

#endif
