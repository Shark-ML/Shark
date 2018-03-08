/*!
 * \brief       Kernels for folding matrix expressions
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
#ifndef REMORA_KERNELS_DEFAULT_MATRIX_FOLD_HPP
#define REMORA_KERNELS_DEFAULT_MATRIX_FOLD_HPP

#include "../../detail/traits.hpp" //orientations
#include "../../expression_types.hpp"
#include <type_traits>
namespace remora{namespace bindings{
	
template<class F, class M, class Orientation, class Tag>
void matrix_fold(matrix_expression<M, cpu_tag> const& m, typename F::result_type& value, Orientation, Tag) {
	F f;
	for(std::size_t i = 0; i != major_size(m); ++i){
		auto end = m().major_end(i);
		for(auto pos = m().major_begin(i);pos != end; ++pos){
			value = f(value,*pos);
		}
	}
}

}}
#endif
