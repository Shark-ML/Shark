/*!
 * \brief       kernels for folding kernels with openCL
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
#ifndef REMORA_KERNELS_CLBLAS_VECTOR_FOLD_HPP
#define REMORA_KERNELS_CLBLAS_VECTOR_FOLD_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <boost/compute/container/array.hpp>
#include <boost/compute/algorithm/copy_n.hpp>
namespace remora{namespace bindings{

template<class F, class V>
void vector_fold(vector_expression<V, gpu_tag> const& v_unreg, typename F::result_type& value, dense_tag) {
	if(v_unreg().size() == 0) return;
	auto& queue = v_unreg().queue();
	typedef typename F::result_type value_type;
	gpu::detail::meta_kernel k("blas_vector_fold");
	std::size_t size_index = k.add_arg<std::size_t>("size");
	auto v = k.register_args(to_functor(v_unreg));
	auto f = k.register_args(F());
	
	boost::compute::array<value_type,1> device_result;
	boost::compute::copy_n(&value, 1, device_result.begin(), queue);
	auto exprSubFold = k.expr<value_type>("subfold[get_local_id(0)]");
	k << "__local " <<k.decl<value_type>("subfold")<< "[TILE_DIM];\n";
	k << exprSubFold<<" = "<<v(k.expr<cl_uint>("min(size-1,get_local_id(0))"))<<";\n";
	k << "for(uint i = TILE_DIM + get_local_id(0); i < size; i += TILE_DIM){\n    ";
	k << exprSubFold << '=' << f(exprSubFold,v(k.expr<cl_uint>("i")))<<";\n";
	k << "}\n";
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";//wait until all threads are done with computing
	//sum up the rows
	k << "if(get_local_id(0) == 0){\n";
	k << "    for(uint i = 1 ; i < min((uint)size,(uint)TILE_DIM); ++i){\n";
	k << "        subfold[0] =" << f(k.expr<value_type>("subfold[0]"),k.expr<value_type>("subfold[i]"))<<";\n";
	k << "    }\n    ";
	k << device_result.begin()[0]<< "= subfold[0];\n";
	k << "}\n";
	
	std::size_t TILE_DIM = 32;
	boost::compute::kernel kernel = k.compile(queue.get_context(), "-DTILE_DIM=32");
	kernel.set_arg(size_index, v_unreg().size());
	
	std::size_t global_work_size[1] = {TILE_DIM};
	std::size_t local_work_size[1] = {TILE_DIM};
	queue.enqueue_nd_range_kernel(kernel, 1,nullptr, global_work_size, local_work_size);
	boost::compute::copy_n(device_result.begin(), 1, &value, queue);
}


}}
#endif
