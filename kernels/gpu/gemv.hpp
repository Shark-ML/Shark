//===========================================================================
/*!
 * 
 *
 * \brief       -
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
#ifndef REMORA_KERNELS_GPU_GEMV_HPP
#define REMORA_KERNELS_GPU_GEMV_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <boost/compute/functional/operator.hpp> //for multiplies

namespace remora{ namespace kernels{

// v <- v + alpha * A * x
template <typename MatA, typename VecX, typename VecV>
void gemv(
	matrix_expression<MatA, gpu_tag> const& A_unreg,
	vector_expression<VecX, gpu_tag> const& x_unreg,
	vector_expression<VecV, gpu_tag>& v_unreg, 
	typename VecV::value_type const& alpha
) {
	REMORA_SIZE_CHECK(A_unreg().size1() == v_unreg().size());
	REMORA_SIZE_CHECK(A_unreg().size2() == x_unreg().size());
	
	
	typedef typename VecV::value_type value_type;
	boost::compute::multiplies<value_type> prod;
	gpu::detail::meta_kernel k("blas_gemv");
	std::size_t alpha_index = k.add_arg<value_type>("alpha");
	std::size_t size1_index = k.add_arg<std::size_t>("size1");
	std::size_t size2_index = k.add_arg<std::size_t>("size2");
	auto A = k.register_args(to_functor(A_unreg));
	auto x = k.register_args(to_functor(x_unreg));
	auto v = k.register_args(to_functor(v_unreg));
	//read all tiles in the assigned rows and compute the inner product
	k << "__local " <<k.decl<value_type>("results")<< "[TILE_DIM][TILE_DIM+2];";
	k << "uint rowid = get_global_id(0);";
	k << "results[get_local_id(0)][get_local_id(1)]  = 0.0;";
	k << "for(uint i = get_local_id(1) ; i < size2 && rowid < size1; i += TILE_DIM){";
	auto exprRow = k.expr<cl_uint>("rowid");
	auto exprCol = k.expr<cl_uint>("i");
	k<< "    results[get_local_id(0)][get_local_id(1)] += "<< prod(A(exprRow,exprCol),x(exprCol))<<";";
	k<<'}';
	k << "barrier(CLK_LOCAL_MEM_FENCE);";//wait until all threads are done with computing
	//sum up the rows
	k << "if(get_local_id(1) == 0 && rowid < size1){";
	k << "    for(uint i = 1 ; i < TILE_DIM; ++i){";
	k << "        results[get_local_id(0)][0] +=results[get_local_id(0)][i];";
	k << "    }";
	k << v(exprRow) << "+= alpha * results[get_local_id(0)][0];";
	k<< "}";
	//create source

	std::size_t TILE_DIM = 16;
	char const* options ="-DTILE_DIM=16";
	boost::compute::kernel kernel = k.compile(v_unreg().queue().get_context(), options);
	//enqueue kernel
	kernel.set_arg(alpha_index, alpha);
	kernel.set_arg(size1_index, A_unreg().size1());
	kernel.set_arg(size2_index, A_unreg().size2());
	
	std::size_t global_work_size[2] = {
		((A_unreg().size1()+TILE_DIM-1)/TILE_DIM) * TILE_DIM,
		TILE_DIM
	};
	std::size_t local_work_size[2] = {TILE_DIM,TILE_DIM};
	v_unreg().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, local_work_size);
}

}}

#endif
