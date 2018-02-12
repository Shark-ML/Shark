/*!
 * \brief       Assignment kernels for vector expressions
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
#ifndef REMORA_KERNELS_CLBLAS_VECTOR_ASSIGN_HPP
#define REMORA_KERNELS_CLBLAS_VECTOR_ASSIGN_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"

namespace remora{namespace bindings{
	
template<class F, class V>
void apply(vector_expression<V, gpu_tag>& v_unreg, F const& f_unreg) {
	if(v_unreg().size() == 0) return;
	gpu::detail::meta_kernel k("blas_vector_apply_dense");
	
	auto v = k.register_args(to_functor(v_unreg));
	auto f = k.register_args(f_unreg);
	
	//create source
	k<<v(k.get_global_id(0))<<" = " << f(v(k.get_global_id(0)))<<";";
	boost::compute::kernel kernel = k.compile(v_unreg().queue().get_context());
	//enqueue kernel
	std::size_t global_work_size[1] = {v_unreg().size()};
	v_unreg().queue().enqueue_nd_range_kernel(kernel, 1,nullptr, global_work_size, nullptr);
}

template<class F, class V>
void assign(vector_expression<V, gpu_tag>& v, typename V::value_type t) {
	static_assert(std::is_base_of<dense_tag, typename V::storage_type::storage_tag>::value, "target must have dense storage for assignment");
	auto f = device_traits<gpu_tag>::make_bind_second(F(), t);
	apply(v,f);
}

////////////////////////////////////////////
//assignment with functor
////////////////////////////////////////////

// Dense-Dense case
template<class V, class E, class F>
void vector_assign_functor(
	vector_expression<V, gpu_tag>& v_unreg,
	vector_expression<E, gpu_tag> const& e_unreg,
	F f_unreg,
	dense_tag, dense_tag
) {
	if(v_unreg().size() == 0) return;
	
	gpu::detail::meta_kernel k("blas_vector_assign_functor_dense");
	
	auto v = k.register_args(to_functor(v_unreg));
	auto e = k.register_args(to_functor(e_unreg));
	auto f = k.register_args(f_unreg);
	
	//create source
	k<<v(k.get_global_id(0))<<" = " << f(v(k.get_global_id(0)), e(k.get_global_id(0)))<<";";
	boost::compute::kernel kernel = k.compile(v_unreg().queue().get_context());
	//enqueue kernel
	std::size_t global_work_size[1] = {v_unreg().size()};
	v_unreg().queue().enqueue_nd_range_kernel(kernel, 1,nullptr, global_work_size, nullptr);
}

/////////////////////////////////////////////////////////
//direct assignment of two vectors
////////////////////////////////////////////////////////

// Dense-Dense case
template< class V, class E>
void vector_assign(
	vector_expression<V, gpu_tag>& v, vector_expression<E, gpu_tag> const& e, 
	dense_tag t, dense_tag
) {
	vector_assign_functor(v, e, device_traits<gpu_tag>::right_arg<typename E::value_type>(), t, t);
}




}}
#endif
