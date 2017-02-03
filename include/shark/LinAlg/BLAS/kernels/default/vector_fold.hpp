/*!
 * \brief       Kernels for folding vector expressions
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
#ifndef REMORA_KERNELS_DEFAULT_VECTOR_FOLD_HPP
#define REMORA_KERNELS_DEFAULT_VECTOR_FOLD_HPP

#include "../../expression_types.hpp"

namespace remora{namespace bindings{
template<class F, class V>
void vector_fold(vector_expression<V, cpu_tag> const& v, typename F::result_type& value, dense_tag) {
	F f;
	std::size_t size = v().size();
	for(std::size_t i = 0; i != size; ++i){
		value = f(value,v()(i));
	}
}

template<class F, class V>
void vector_fold(vector_expression<V, cpu_tag> const& v, typename F::result_type& value, sparse_tag) {
	F f;
	std::size_t nnz = 0;
	auto iter = v().begin();
	auto end = v().end();
	for(;iter != end;++iter,++nnz){
		value = f(value,*iter);
	}
	//apply final operator f(0,v)
	if(nnz != v().size())
		value = f(value, 0);
}

}}
#endif
