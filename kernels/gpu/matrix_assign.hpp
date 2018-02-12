/*!
 * \brief       Kernels for matrix-expression assignments on the gpu
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
#ifndef REMORA_KERNELS_CLBLAS_MATRIX_ASSIGN_HPP
#define REMORA_KERNELS_CLBLAS_MATRIX_ASSIGN_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"

#include <iostream>
namespace remora{namespace bindings{
	
	
//////////////////////////////////////////////////////
////Apply function elementwise to Matrix
/////////////////////////////////////////////////////

// Explicitly iterating row major
template<class F, class M, class Orientation>
void matrix_apply(
	matrix_expression<M, gpu_tag>& m_unreg, 
	F const& f_unreg,
	Orientation
){
	gpu::detail::meta_kernel k("blas_matrix_apply_dense");
	
	auto m = k.register_args(to_functor(m_unreg));
	auto f = k.register_args(f_unreg);
	
	//create source
	k<<m(k.get_global_id(0),k.get_global_id(1))<<" = " << f(m(k.get_global_id(0),k.get_global_id(1)))<<";";
	boost::compute::kernel kernel = k.compile(m_unreg().queue().get_context());
	//enqueue kernel
	std::size_t global_work_size[2] = {m_unreg().size1(), m_unreg().size2()};
	m_unreg().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, nullptr);
}
	
//////////////////////////////////////////////////////
////Scalar Assignment to Matrix
/////////////////////////////////////////////////////

// Explicitly iterating row major
template<class F, class M, class Orientation>
void matrix_assign(
	matrix_expression<M, gpu_tag>& m, 
	typename M::value_type t, 
	Orientation o
){
	static_assert(std::is_base_of<dense_tag, typename M::storage_type::storage_tag>::value, "target must have dense storage for assignment");
	auto f = device_traits<gpu_tag>::make_bind_second(F(), t);
	matrix_apply(m,f,o);
}


///////////////////////////////////////////////////////////////////////////////////////////
//////Matrix Assignment With Functor implementing +=,-=...
///////////////////////////////////////////////////////////////////////////////////////////

template<class F, class M, class E>
void matrix_assign_functor(
	matrix_expression<M, gpu_tag>& m_unreg, 
	matrix_expression<E, gpu_tag> const& e_unreg,
	F f_unreg,
	row_major, row_major ,dense_tag, dense_tag
){
	//create source
	gpu::detail::meta_kernel k("blas_matrix_assign");
	auto m = k.register_args(to_functor(m_unreg));
	auto e = k.register_args(to_functor(e_unreg));
	auto f = k.register_args(f_unreg);

	auto id0 = k.expr<cl_uint>("get_global_id(0)");
	auto id1 = k.expr<cl_uint>("get_global_id(1)");
	k<< m(id0,id1) << "=" << f(m(id0,id1),e(id0,id1))<<";\n";
	//enqueue kernel
	boost::compute::kernel kernel = k.compile(m_unreg().queue().get_context());
	std::size_t global_work_size[2] = {m_unreg().size1(), m_unreg().size2()};
	m_unreg().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, nullptr);
}

//dense-dense case row-major, column-major
template<class F,class M, class E>
void matrix_assign_functor(
	matrix_expression<M, gpu_tag>& m_unreg, 
	matrix_expression<E, gpu_tag> const& e_unreg,
	F f_unreg,
	row_major, column_major ,dense_tag, dense_tag
) {
	typedef typename M::value_type value_type;
	std::size_t TILE_DIM = 32;
	char const* options ="-DTILE_DIM=32ul";
	//There are usually not enough parallel worker threads in a local group
	//to fill a tile. Therefore every parallel threads reads several elements.
	//BLOCK_COLS are the number of parallel threads reading a column
	//and must be a divisor of TILE_DIM
	std::size_t BLOCK_COLS = 8; 
	
	//create source
	gpu::detail::meta_kernel k("blas_matrix_assign_row_col");
	auto m = k.register_args(to_functor(m_unreg));
	auto e = k.register_args(to_functor(e_unreg));
	auto f = k.register_args(f_unreg);
	//create local memory. we first copy a tile in local
	// memory which gets the orientation right. Then we copy the tile
	//to the target
	// TILE_DIM+1 is here to avoid bank conflicts in local memory
	std::size_t size1_index = k.add_arg<std::size_t>("size1");
	std::size_t size2_index = k.add_arg<std::size_t>("size2");
	k << "__local " <<k.decl<value_type>("tile")<< "[TILE_DIM][TILE_DIM+2];\n";
	k << "uint base_row = get_group_id(0) * TILE_DIM;\n";
	k << "uint base_col = get_group_id(1) * TILE_DIM;\n";
	//copy indices, into local memory, note the change of direction
	//also note that if we are on the boundary, the tile
	// might be largerthan the amount of values to read
	k << "uint maxDim1 = min(size1-base_row,TILE_DIM);\n";
	k << "uint maxDim2 = min(size2-base_col,TILE_DIM);\n";
	k << "for(uint i = get_local_id(1) ; i < maxDim2 && get_local_id(0) < maxDim1; i += get_local_size(1)){\n";
	auto row_exp = k.expr<cl_uint>("(base_row+get_local_id(0))");
	auto col_exp = k.expr<cl_uint>("(base_col+i)");
	k << "    tile[get_local_id(0)][i] =" << e(row_exp, col_exp)<<";\n";
	k << "}\n";
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";//wait until all threads are done with copying
	// write output from local memory, again be sure that 
	// we do not write outside the feasible area
	k << "for(uint i = get_local_id(1); i < maxDim1 && get_local_id(0) < maxDim2; i += get_local_size(1)){\n"; 
	auto target = m(k.expr<cl_uint>("(base_row + i)"), k.expr<cl_uint>("(base_col + get_local_id(0))"));
	k << target << " = " <<f(target, k.expr<cl_uint>("tile[i][get_local_id(0)]"))<<";\n";
	k << "}\n";
	
	//compile kernel
	
	boost::compute::kernel kernel = k.compile(m_unreg().queue().get_context(), options);
	
	//enqueue kernel
	kernel.set_arg(size1_index, m_unreg().size1());
	kernel.set_arg(size2_index, m_unreg().size2());
	std::size_t global_work_size[2] = {(m_unreg().size1()+TILE_DIM-1) / TILE_DIM * TILE_DIM, (m_unreg().size2()+TILE_DIM-1) / TILE_DIM * BLOCK_COLS };
	std::size_t local_work_size[2] = {TILE_DIM, BLOCK_COLS};
	m_unreg().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, local_work_size);
}

/////////////////////////////////////////////////////////////////
//////Matrix Assignment implementing op=
////////////////////////////////////////////////////////////////

template<class M, class E>
void matrix_assign(
	matrix_expression<M, gpu_tag> &m, 
	matrix_expression<E, gpu_tag> const& e,
	row_major o, row_major,dense_tag t, dense_tag
) {
	matrix_assign_functor(m, e, device_traits<gpu_tag>::right_arg<typename E::value_type>(), o, o, t, t);
}

//dense-dense case
template<class M, class E>
void matrix_assign(
	matrix_expression<M, gpu_tag> &m, 
	matrix_expression<E, gpu_tag> const& e,
	row_major o1, column_major o2,dense_tag t, dense_tag
) {
	matrix_assign_functor(m, e, device_traits<gpu_tag>::right_arg<typename E::value_type>(), o1, o2, t, t);
}


}}

#endif
