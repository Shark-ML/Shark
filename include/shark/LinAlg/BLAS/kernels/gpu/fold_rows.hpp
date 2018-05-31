/*!
 * 
 *
 * \brief       Folds the rows of a row-major or column major matrix.
 *
 * \author      O. Krause
 * \date        2018
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

#ifndef REMORA_KERNELS_CLBLAS_FOLD_ROWS_HPP
#define REMORA_KERNELS_CLBLAS_FOLD_ROWS_HPP

#include "../../expression_types.hpp"
#include "../../detail/traits.hpp"
#include <boost/compute/functional/operator.hpp> //for multiplies

namespace remora{namespace bindings{

template<class F, class G, class M, class V, class Orientation>
void fold_rows(
	matrix_expression<M, gpu_tag> const& A_unreg, 
	vector_expression<V, gpu_tag>& v_unreg,
	F f_unreg,
	G g_unreg,
	Orientation
){
	typedef typename V::value_type value_type;
	gpu::detail::meta_kernel k("remora_fold_rows");
	std::size_t size1_index = k.add_arg<std::size_t>("size1");
	std::size_t size2_index = k.add_arg<std::size_t>("size2");
	auto A = k.register_args(to_functor(A_unreg));
	auto v = k.register_args(to_functor(v_unreg));
	auto f = k.register_args(f_unreg);
	auto g = k.register_args(g_unreg);
	//read all tiles in the assigned rows and sum them up
	k << "__local " <<k.decl<value_type>("folds")<< "[TILE_DIM][TILE_DIM+1];\n";
	k << "ulong rowid = get_global_id(0);\n";
	k << "ulong colid = get_global_id(1);\n";
	k << "if(rowid < size1 && colid < size2){\n"; //can not compute rows/columns that are infeasible
	//note: we can not simply step out here as we must ensure that all threads get to the barrier(...)
	auto colid = k.expr<cl_ulong>("colid");
	auto rowid = k.expr<cl_ulong>("rowid");
	auto entry = k.expr<cl_ulong>("folds[get_local_id(0)][get_local_id(1)]");
	k << "	"<<entry <<" = "<< A(rowid,colid) <<";\n";
	k << "	colid += TILE_DIM;\n";
	k << "	for(; colid < size2; colid += TILE_DIM){\n";
	k << "		"<< entry << " = " << f(entry, A(rowid,colid))<<";\n";
	k << "	}\n";
	k << "}\n";
	k << "barrier(CLK_LOCAL_MEM_FENCE);\n";//wait until all threads are done with folding the columns
	//final fold, just the threads in the first row compute this
	k << "if(get_local_id(1) == 0 && rowid < size1){\n";
	k << "    for(uint i = 1 ; i < min(TILE_DIM, size2); ++i){\n";
	k << "        " << entry <<" = "<< f(entry, k.expr<cl_ulong>("folds[get_local_id(0)][i]"))<<";\n";
	k << "    }\n";
	k << v(rowid) << "+= " <<g(k.expr<value_type>("folds[get_local_id(0)][0]"))<<";\n";
	k<< "}\n";
	//create source

	std::size_t TILE_DIM = 8;
	char const* options ="-DTILE_DIM=8ul";
	boost::compute::kernel kernel = k.compile(v_unreg().queue().get_context(), options);
	//enqueue kernel
	kernel.set_arg(size1_index, A_unreg().size1());
	kernel.set_arg(size2_index, A_unreg().size2());
	
	std::size_t global_size[2] = {
		((A_unreg().size1()+TILE_DIM-1)/TILE_DIM) * TILE_DIM,
		TILE_DIM
	};
	std::size_t local_size[2] = {TILE_DIM, TILE_DIM};
	v_unreg().queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_size, local_size);
}


}}

#endif
