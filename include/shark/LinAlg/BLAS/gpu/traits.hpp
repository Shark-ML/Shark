//===========================================================================
/*!
 * 
 *
 * \brief       Traits of gpu expressions
 *
 * \author      O. Krause
 * \date        2016
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

#ifndef SHARK_LINALG_BLAS_GPU_DETAIL_TRAITS_HPP
#define SHARK_LINALG_BLAS_GPU_DETAIL_TRAITS_HPP

#include "../detail/traits.hpp"
#include <boost/compute/core.hpp>

namespace shark{namespace blas{namespace gpu{
	
template<class T>
struct dense_vector_storage{
	typedef dense_tag storage_tag;
	boost::compute::vector<T> const& buffer;
	std::size_t offset;
	std::size_t stride;
	
	dense_vector_storage<T> sub_region(std::size_t offset){
		return {buffer, this->offset+offset, stride};
	}
};

template<class T>
struct dense_matrix_storage{
	typedef dense_tag storage_tag;
	typedef dense_vector_storage<T> row_storage;
	boost::compute::vector<T> const& buffer;
	std::size_t offset;
	std::size_t leading_dimension;
	
	template<class Orientation>
	dense_matrix_storage<T> sub_region(std::size_t offset1, std::size_t offset2, Orientation){
		std::size_t offset_major = Orientation::index_M(offset1,offset2);
		std::size_t offset_minor = Orientation::index_m(offset1,offset2);
		return {buffer, offset + offset_major*leading_dimension+offset_minor, leading_dimension};
	}
	
	template<class Orientation>
	row_storage row(std::size_t i, Orientation){
		return {buffer, offset + i * Orientation::index_M(leading_dimension,1), Orientation::index_m(leading_dimension,1)};
	}
	template<class Orientation>
	row_storage diag(){
		return {buffer, offset, leading_dimension+1};
	}
};

}}}

#endif