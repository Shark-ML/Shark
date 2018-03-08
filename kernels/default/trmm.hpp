/*!
 *
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2010
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

#ifndef REMORA_KERNELS_DEFAULT_TRMM_HPP
#define REMORA_KERNELS_DEFAULT_TRMM_HPP

#include "../../expression_types.hpp"//for matrix_expression
#include "../../proxy_expressions.hpp"//proxies for blocking
#include "../../detail/traits.hpp"//triangular_tag
#include "simd.hpp"
#include "mgemm.hpp" //block macro kernel for dense syrk
#include <type_traits> //std::false_type marker for unoptimized, std::common_type
namespace remora{ namespace bindings {

template <typename T>
struct trmm_block_size {
	typedef detail::block<T> block;
	static const unsigned mr = 4; // stripe width for lhs
	static const unsigned nr = 3 * block::max_vector_elements; // stripe width for rhs
	static const unsigned lhs_block_size = 32 * mr;
	static const unsigned rhs_column_size = (1024 / nr) * nr;
	static const unsigned align = 64; // align temporary arrays to this boundary
};

template <class E, class T, class block_size, bool unit>
void pack_A_triangular(matrix_expression<E, cpu_tag> const& A, T* p, block_size, triangular_tag<false,unit>){
	BOOST_ALIGN_ASSUME_ALIGNED(p, block_size::block::align);

	std::size_t const mc = A().size1();
	std::size_t const kc = A().size2();
	static std::size_t const MR = block_size::mr;
	const std::size_t mp = (mc+MR-1) / MR;

	std::size_t nu = 0;
	for (std::size_t l=0; l<mp; ++l) {
		for (std::size_t j=0; j<kc; ++j) {
			for (std::size_t i = l*MR; i < l*MR + MR; ++i,++nu) {
				if(unit && i == j)
					p[nu] = 1.0;
				else
					p[nu] = ((i<mc) && (i >= j)) ? A()(i,j) : T(0);
			}
		}
	}
}

template <class E, class T, class block_size, bool unit>
void pack_A_triangular(matrix_expression<E, cpu_tag> const& A, T* p, block_size, triangular_tag<true,unit>){
	BOOST_ALIGN_ASSUME_ALIGNED(p, block_size::block::align);

	std::size_t const mc = A().size1();
	std::size_t const kc = A().size2();
	static std::size_t const MR = block_size::mr;
	const std::size_t mp = (mc+MR-1) / MR;

	std::size_t nu = 0;
	for (std::size_t l=0; l<mp; ++l) {
		for (std::size_t j=0; j<kc; ++j) {
			for (std::size_t i = l*MR; i < l*MR + MR; ++i,++nu) {
				if(unit && i == j)
					p[nu] = 1.0;
				else
					p[nu] = ((i<mc) && (i <= j)) ? A()(i,j) : T(0);
			}
		}
	}
}


template <class E1, class E2, class Triangular>
void trmm_impl(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> & e2,
	Triangular t
){
	typedef typename std::common_type<typename E1::value_type, typename E2::value_type>::type value_type;
	typedef trmm_block_size<value_type> block_size;

	static const std::size_t AC = block_size::lhs_block_size;
	static const std::size_t BC = block_size::rhs_column_size;

	//obtain uninitialized aligned storage
	boost::alignment::aligned_allocator<value_type,block_size::block::align> allocator;
	value_type* A = allocator.allocate(AC * AC);
	value_type* B = allocator.allocate(AC * BC);

	//figure out number of blocks to use
	const std::size_t  M = e1().size1();
	const std::size_t  N = e2().size2();
	const std::size_t Ab = (M+AC-1) / AC;
	const std::size_t Bb = (N+BC-1) / BC;

	//get access to raw storage of B
	auto storageB = e2().raw_storage();
	auto Bpointer = storageB.values;
	const std::size_t stride1 = E2::orientation::index_M(storageB.leading_dimension,1);
	const std::size_t stride2 = E2::orientation::index_m(storageB.leading_dimension,1);

	for (std::size_t j = 0; j < Bb; ++j) {//columns of B
		std::size_t nc = std::min(BC, N - j * BC);
		//We have to make use of the triangular nature of B and the fact that rhs and storage are the same matrix
		//for upper triangular matrices we can compute whole columns of A starting from the first
		//until we reach the block on the diagonal, where we stop.
		//for lower triangular matrices we start with the last column instead.
		for (std::size_t k0 = 0; k0< Ab; ++k0){//row-blocks of B = column blocks of A.
			std::size_t k = Triangular::is_upper? k0: Ab-k0-1;
			std::size_t kc = std::min(AC, M - k * AC);
			//read block of B
			auto Bs = subrange(e2, k*AC, k*AC + kc, j * BC, j * BC + nc );
			pack_B_dense(Bs, B, block_size());
			Bs.clear();//its going to be overwritten with the result

			//apply block of B to all blocks of A needing it. we will overwrite the real B block as a result of this.
			//this is okay as the block is stored as argument
			std::size_t i_start = Triangular::is_upper? 0: k;
			std::size_t i_end = Triangular::is_upper? k+1: Ab;
			for (std::size_t i = i_start; i < i_end; ++i) {
				std::size_t mc = std::min(AC, M - i * AC);
				auto As = subrange(e1, i * AC, i * AC + mc, k * AC, k * AC + kc);
				if(i == k){
					pack_A_triangular(As, A, block_size(), t);
				}else
					pack_A_dense(As, A, block_size());

				mgemm(
					mc, nc, kc, value_type(1.0), A, B,
					&Bpointer[i*AC * stride1 + j*BC * stride2], stride1, stride2, block_size()
				);
			}
		}
	}
	//free storage
	allocator.deallocate(A,AC * AC);
	allocator.deallocate(B,AC * BC);
}



//main kernel runs the kernel above recursively and calls gemv
template <bool Upper,bool Unit,typename MatA, typename MatB>
void trmm(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag>& B,
	std::false_type //unoptimized
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size2() == B().size1());

	trmm_impl(A,B, triangular_tag<Upper,Unit>());
}

}}

#endif
