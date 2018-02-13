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

#ifndef REMORA_KERNELS_DEFAULT_SYRK_HPP
#define REMORA_KERNELS_DEFAULT_SYRK_HPP

#include "../../expression_types.hpp"//for matrix_expression
#include "../../proxy_expressions.hpp"//for matrix range/transpose
#include "mgemm.hpp" //block macro kernel for dense syrk
#include <type_traits> //std::false_type marker for unoptimized, std::common_Type

namespace remora { namespace bindings {


template <typename T>
struct syrk_block_size {
	typedef detail::block<T> block;
	static const unsigned mr = 4; // stripe width for E_left
	static const unsigned nr = mr * block::max_vector_elements; // stripe width for E_right
	static const unsigned lhs_block_size = 3 * mr * nr;//square block size of M to compute
	static const unsigned rhs_k_size = 1024;//strip of ks to compute
};
template <class E, class Mat, class Triangular>
void syrk_impl(
	matrix_expression<E, cpu_tag> const& e,
	matrix_expression<Mat, cpu_tag>& m,
	typename Mat::value_type& alpha,
	Triangular t
){
	typedef typename E::value_type value_type;
	typedef syrk_block_size<value_type> block_size;

	static const std::size_t MC = block_size::lhs_block_size;
	static const std::size_t EC = block_size::rhs_k_size;

	//obtain uninitialized aligned storage
	boost::alignment::aligned_allocator<value_type,block_size::block::align> allocatorE;
	boost::alignment::aligned_allocator<typename Mat::value_type,block_size::block::align> allocatorM;
	value_type* E_left = allocatorE.allocate(MC * EC);
	value_type* E_right = allocatorE.allocate(MC * EC);
	auto M_diagonal_block = allocatorM.allocate(MC * MC);

	//figure out number of blocks to use
	const std::size_t  M = e().size1();
	const std::size_t  K = e().size2();
	const std::size_t Mb = (M+MC-1) / MC;//we split m in Mb x Mb blocks
	const std::size_t Eb = (K+EC-1) / EC;//we split B in Mb x Eb blocks

	//get access to raw storage of M
	auto storageM = m().raw_storage();
	auto Mpointer = storageM.values;
	const std::size_t stride1 = Mat::orientation::index_M(storageM.leading_dimension,1);
	const std::size_t stride2 = Mat::orientation::index_m(storageM.leading_dimension,1);

	for (std::size_t k = 0; k < Eb; ++k) {//column blocks of E
		std::size_t kc = std::min(EC, K - k * EC);
		for (std::size_t i = 0; i < Mb; ++i){//row-blocks of M
			std::size_t mc = std::min(MC, M - i * MC);
			//load block of the left E into memory
			auto E_lefts = subrange(e, i * MC, i * MC + mc, k*EC, k*EC + kc );
			pack_A_dense(E_lefts, E_left, block_size());

			std::size_t start_j = Triangular::is_upper? i : 0;
			std::size_t end_j = Triangular::is_upper? Mb : i+1;
			for(std::size_t j = start_j; j < end_j; ++j){//traverse over the blocks that are to be computed
				std::size_t mc2 = std::min(MC, M - j * MC);
				//load block of the right E into memory
				auto E_rights = subrange(e, j * MC, j * MC + mc2, k*EC, k*EC + kc );
				auto E_rights_trans = trans(E_rights);
				//~ auto E_rights_trans = trans(subrange(e, j * MC, j * MC + mc2, k*EC, k*EC + kc ));
				pack_B_dense(E_rights_trans, E_right, block_size());

				if(i==j){//diagonal block: we have to ensure that we only access elements on the diagonal
					for(std::size_t i0 = 0; i0 != mc; ++i0){
						for(std::size_t j0 = 0; j0 != mc2; ++j0){
							M_diagonal_block[i0*MC+j0] = 0.0;
						}
					}
					mgemm(
						mc, mc2, kc, alpha, E_left, E_right,
						M_diagonal_block, MC, 1, block_size()
					);
					auto M_diagonal = Mpointer + i * MC * stride1 + j * MC * stride2;
					for(std::size_t i0 = 0; i0 != mc; ++i0){
						std::size_t start_j0 = Triangular::is_upper? i0 : 0;
						std::size_t end_j0 = Triangular::is_upper? mc2 : i0+1;
						for(std::size_t j0 = start_j0; j0 < end_j0; ++j0){
							M_diagonal[i0*stride1+j0*stride2] += M_diagonal_block[i0*MC+j0];
						}
					}
				}else{
					mgemm(
						mc, mc2, kc, alpha, E_left, E_right,
						&Mpointer[i*MC * stride1 + j*MC * stride2], stride1, stride2, block_size()
					);
				}
			}
		}
	}
	//free storage
	allocatorE.deallocate(E_left,MC * EC);
	allocatorE.deallocate(E_right,MC * EC);
	allocatorM.deallocate(M_diagonal_block, MC * MC);
}



//main kernel runs the kernel above recursively and calls gemv
template <bool Upper, typename M, typename E>
void syrk(
	matrix_expression<E, cpu_tag> const& e,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type& alpha,
	std::false_type //unoptimized
){
	REMORA_SIZE_CHECK(m().size1() == m().size2());
	REMORA_SIZE_CHECK(m().size2() == e().size1());

	syrk_impl(e,m, alpha, triangular_tag<Upper,false>());
}

}}

#endif
