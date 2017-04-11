/*!
 *
 *
 * \brief       dense matrix matrix multiplication implementation
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

#ifndef REMORA_KERNELS_DEFAULT_DENSE_GEMM_HPP
#define REMORA_KERNELS_DEFAULT_DENSE_GEMM_HPP

#include "../gemv.hpp"//for dispatching to gemv
#include "../../assignment.hpp"//plus_assign
#include "../../detail/matrix_proxy_classes.hpp"//matrix row,column,transpose,range
#include "mgemm.hpp" //block macro kernel for dense gemm
#include <type_traits> //std::common_type


namespace remora{namespace bindings {

//  Dense Block-GEMM implementation based on boost.ublas
//  written by:
//  Copyright (c) 2016
//  Michael Lehn, Imre Palik
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

template <typename T>
struct gemm_block_size {
	typedef detail::block<T> block;
	static const unsigned mr = 4; // stripe width for lhs
	static const unsigned nr = 3 * block::max_vector_elements; // stripe width for rhs
	static const unsigned mc = 128;
	static const unsigned kc = 512; // stripe length
	static const unsigned nc = (1024/nr) * nr;
};

template <>
struct gemm_block_size<float> {
	typedef detail::block<float> block;
	static const unsigned mc = 256;
	static const unsigned kc = 512; // stripe length
	static const unsigned nc = 4096;
	static const unsigned mr = 4; // stripe width for lhs
	static const unsigned nr = 16; // stripe width for rhs
};

template <>
struct gemm_block_size<long double> {
	typedef detail::block<long double> block;
	static const unsigned mc = 256;
	static const unsigned kc = 512; // stripe length
	static const unsigned nc = 4096;
	static const unsigned mr = 1; // stripe width for lhs
	static const unsigned nr = 4; // stripe width for rhs
};

//-- Dense gemm
template <class E1, class E2, class Mat>
void dense_gemm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<Mat, cpu_tag>& m,
	typename Mat::value_type alpha
){
	static_assert(std::is_same<typename Mat::orientation,row_major>::value,"target matrix must be row major");
	typedef typename std::common_type<
		typename E1::value_type, typename E2::value_type, typename Mat::value_type
	>::type value_type;

	typedef gemm_block_size<
		typename std::common_type<typename E1::value_type, typename E2::value_type>::type
	> block_size;

	static const std::size_t MC = block_size::mc;
	static const std::size_t NC = block_size::nc;
	static const std::size_t KC = block_size::kc;

	//obtain uninitialized aligned storage
	boost::alignment::aligned_allocator<value_type,block_size::block::align> allocator;
	value_type* A = allocator.allocate(MC * KC);
	value_type* B = allocator.allocate(NC * KC);

	const std::size_t M = m().size1();
	const std::size_t N = m().size2();
	const std::size_t K = e1().size2 ();
	const std::size_t mb = (M+MC-1) / MC;
	const std::size_t nb = (N+NC-1) / NC;
	const std::size_t kb = (K+KC-1) / KC;

	auto storageM = m().raw_storage();
	auto C_ = storageM.values;
	const std::size_t ldc = storageM.leading_dimension;
	for (std::size_t j=0; j<nb; ++j) {
		std::size_t nc = std::min(NC, N - j*NC);

		for (std::size_t l=0; l<kb; ++l) {
			std::size_t kc = std::min(KC, K - l*KC);
			matrix_range<typename const_expression<E2>::type> Bs(e2(), l*KC, l*KC+kc, j*NC, j*NC+nc);
			pack_B_dense(Bs, B, block_size());

			for (std::size_t i=0; i<mb; ++i) {
				std::size_t mc = std::min(MC, M - i*MC);
				matrix_range<typename const_expression<E1>::type> As(e1(), i*MC, i*MC+mc, l*KC, l*KC+kc);
				pack_A_dense(As, A, block_size());

				mgemm(
					mc, nc, kc, alpha, A, B,
					&C_[i*MC*ldc+j*NC], ldc , 1, block_size()
				);
			}
		}
	}
	//free storage
	allocator.deallocate(A,MC * KC);
	allocator.deallocate(B,NC * KC);
}

}}
#endif
