/*!
 * \brief       Assignment kernels for vector expressions
 * 
 * \author      O. Krause
 * \date        2015
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
#ifndef REMORA_KERNELS_VECTOR_ASSIGN_HPP
#define REMORA_KERNELS_VECTOR_ASSIGN_HPP

#include "../detail/traits.hpp"
#include "default/vector_assign.hpp"
#ifdef REMORA_USE_GPU
#include "gpu/vector_assign.hpp"
#endif

namespace remora{namespace kernels {

	
template<class V, class F, class Device>
void apply(vector_expression<V, Device>& v,F const& f) {
	bindings::apply(v,f);
}
template<class F, class V, class Device>
void assign(vector_expression<V, Device>& v, typename V::value_type t) {
	bindings::assign<F>(v,t);
}

/////////////////////////////////////////////////////////
//direct assignment of two vectors
////////////////////////////////////////////////////////

//dispatcher
template< class V, class E, class Device>
void assign(vector_expression<V, Device>& v, vector_expression<E, Device> const& e) {
	REMORA_SIZE_CHECK(v().size() == e().size());
	typedef typename V::evaluation_category::tag TagV;
	typedef typename E::evaluation_category::tag TagE;
	bindings::vector_assign(v, e,TagV(),TagE());
}

////////////////////////////////////////////
//assignment with functor
////////////////////////////////////////////


// Dispatcher
template<class F, class V, class E, class Device>
void assign(vector_expression<V, Device>& v, vector_expression<E, Device> const& e, F f) {
	REMORA_SIZE_CHECK(v().size() == e().size());
	typedef typename V::evaluation_category::tag TagV;
	typedef typename E::evaluation_category::tag TagE;
	bindings::vector_assign_functor(v(), e(), f, TagV(),TagE());
}

}}
#endif
