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
#include <type_traits>

namespace remora{
	
struct unknown_storage{
	typedef unknown_tag storage_tag;
	template<class O>
	struct row_storage{typedef unknown_storage type;};
	
	typedef unknown_storage diag_storage;
	typedef unknown_storage sub_region_storage;
};

template<class T, class Tag>
struct dense_vector_storage{
	typedef Tag storage_tag;
	T* values;
	std::size_t stride;
	
	dense_vector_storage(){}
	dense_vector_storage(T* values, std::size_t stride):values(values),stride(stride){}
	template<class U, class Tag2>
	dense_vector_storage(dense_vector_storage<U, Tag2> const& storage):
	values(storage.values), stride(storage.stride){
		static_assert(!(std::is_same<Tag,continuous_dense_tag>::value && std::is_same<Tag2,dense_tag>::value), "Trying to assign dense to continuous dense storage");
	}
	
	
	dense_vector_storage<T,Tag> sub_region(std::size_t offset) const{
		return {values+offset*stride, stride};
	}
};

template<class T, class Tag>
struct dense_matrix_storage{
	typedef Tag storage_tag;
	template<class O>
	struct row_storage: public std::conditional<
		std::is_same<O,row_major>::value,
		dense_vector_storage<T, Tag>,
		dense_vector_storage<T, dense_tag>
	>{};
	
	template<class O>
	struct rows_storage: public std::conditional<
		std::is_same<O,row_major>::value,
		dense_matrix_storage<T, Tag>,
		dense_matrix_storage<T, dense_tag>
	>{};
	
	typedef dense_vector_storage<T, dense_tag> diag_storage;
	typedef dense_matrix_storage<T, dense_tag> sub_region_storage;
	
	T* values;
	std::size_t leading_dimension;
	
	dense_matrix_storage(){}
	dense_matrix_storage(T* values, std::size_t leading_dimension):values(values),leading_dimension(leading_dimension){}
	template<class U, class Tag2>
	dense_matrix_storage(dense_matrix_storage<U, Tag2> const& storage):
	values(storage.values), leading_dimension(storage.leading_dimension){
		static_assert(!(std::is_same<Tag,continuous_dense_tag>::value && std::is_same<Tag2,dense_tag>::value), "Trying to assign dense to continuous dense storage");
	}
	
	template<class Orientation>
	sub_region_storage sub_region(std::size_t offset1, std::size_t offset2, Orientation) const{
		std::size_t stride1 = Orientation::index_M(leading_dimension,(std::size_t)1);
		std::size_t stride2 = Orientation::index_m(leading_dimension,(std::size_t)1);
		return {values + offset1 * stride1 + offset2 * stride2, leading_dimension};
	}
	
	template<class Orientation>
	typename rows_storage<Orientation>::type sub_rows(std::size_t offset, Orientation) const{
		std::size_t stride = Orientation::index_M(leading_dimension,(std::size_t)1);
		return {values + offset * stride, leading_dimension};
	}
	
	template<class Orientation>
	typename row_storage<Orientation>::type row(std::size_t i, Orientation) const{
		return {values + i * Orientation::index_M(leading_dimension,(std::size_t)1), Orientation::index_m(leading_dimension,(std::size_t)1)};
	}
	
	diag_storage diag() const{
		return {values, leading_dimension+1};
	}
	
	dense_vector_storage<T, continuous_dense_tag> linear() const{
		return {values, 1};
	}
};


template<class T, class I>
struct sparse_vector_storage{
	typedef sparse_tag storage_tag;
	T* values;
	I* indices;
	std::size_t nnz;
	std::size_t capacity;
	
	sparse_vector_storage(){}
	sparse_vector_storage(T* values, I* indices, std::size_t nnz, std::size_t capacity)
	:values(values), indices(indices), nnz(nnz), capacity(capacity){}
	template<class U, class J>
	sparse_vector_storage(sparse_vector_storage<U, J> const& storage)
	: values(storage.values)
	, indices(storage.indices)
	, nnz(storage.nnz)
	, capacity(storage.capacity){}
};

template<class T>
struct packed_matrix_storage{
	typedef packed_tag storage_tag;
	template<class Orientation>
	struct row_storage{ typedef packed_matrix_storage<T> type;};
	typedef packed_matrix_storage<T> sub_region_storage;
	T* values;
	std::size_t nnz;
};

template<class T, class I>
struct sparse_matrix_storage{
	typedef sparse_tag storage_tag;
	template<class O>
	struct row_storage{ typedef sparse_vector_storage<T,I> type;};
	typedef sparse_matrix_storage<T,I> sub_region_storage;
	T* values;
	I* indices;
	I* major_indices_begin;
	I* major_indices_end;
	std::size_t capacity;
	
	sparse_matrix_storage(
		T* values, I* indices, 
		I* major_indices_begin, I* major_indices_end,
		std::size_t capacity
	):values(values), indices(indices)
	, major_indices_begin(major_indices_begin), major_indices_end(major_indices_end)
	, capacity(capacity){}
	
	template<class U, class J>
	sparse_matrix_storage(sparse_matrix_storage<U, J> const& storage)
	: values(storage.values), indices(storage.indices)
	, major_indices_begin(storage.major_indices_begin), major_indices_end(storage.major_indices_end)
	, capacity(storage.capacity){}
	
	sparse_vector_storage<T,I> row(std::size_t i, row_major)const{
		I start = major_indices_begin[i];
		return {
			values + start, indices + start,
			major_indices_end[i] - start,
			major_indices_begin[i + 1] - start
		};
	}
};
}

#endif
