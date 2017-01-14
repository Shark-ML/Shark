/*!
 *
 *
 * \brief       The mgemm macro kernel used for implementing gemm
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

#ifndef REMORA_KERNELS_DEFAULT_MGEMM_HPP
#define REMORA_KERNELS_DEFAULT_MGEMM_HPP

#include "simd.hpp"
#include <algorithm>//std::fill


namespace remora{namespace bindings {

//  Block-GEMM implementation based on boost.ublas
//  written by:
//  Copyright (c) 2016
//  Michael Lehn, Imre Palik
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//-- Micro Kernel For Dense operations----------------------------------------------------------
template <class block_size, class T, class TC>
void ugemm(
	std::size_t kc, TC alpha, T const* A, T const* B,
	TC* C, std::size_t stride1, std::size_t stride2
){
	BOOST_ALIGN_ASSUME_ALIGNED(A, block_size::block::align);
	BOOST_ALIGN_ASSUME_ALIGNED(B, block_size::block::align);

	typedef typename block_size::block::type vx;
	static const std::size_t vector_length = block_size::block::vector_elements;
	static const std::size_t vecNR = block_size::nr/vector_length;
#ifdef REMORA_USE_SIMD
	vx P[block_size::mr * vecNR] = {};
#else
	typename std::aligned_storage<sizeof(vx[block_size::mr*vecNR]),block_size::block::align>::type Pa;
	T* P = reinterpret_cast<T*>(&Pa);
	for (std::size_t c = 0; c < block_size::mr*vecNR; c++)
		P[c] = 0;
#endif


	// perform the matrix-matrix product as outer product
	// of rows of A and B
	vx const* b = (vx const*)B;
	for (std::size_t l=0; l<kc; ++l) {
		for (std::size_t i=0; i<block_size::mr; ++i) {
			for (std::size_t j=0; j<vecNR; ++j) {
				P[i * vecNR+j] += A[i]*b[j];
			}
		}
		A += block_size::mr;
		b += vecNR;
	}
	//multiply with alpha if necessary
	if (alpha!=TC(1)) {
		for (std::size_t i=0; i<block_size::mr; ++i) {
			for (std::size_t j=0; j< vecNR; ++j) {
				P[i*vecNR+j] *= alpha;
			}
		}
	}

	//add result to C
	T const* p = (T const*) P;
	for (std::size_t i=0; i<block_size::mr; ++i) {
		for (std::size_t j=0; j<block_size::nr; ++j) {
			C[i * stride1+j * stride2] += p[i*block_size::nr+j];
		}
	}
}


// Macro Kernel for two densly packed Blocks
template <class T, class TC, class block_size>
void mgemm(
	std::size_t mc, std::size_t nc, std::size_t kc, TC alpha,
	T const* A, T const* B, TC *C,
	std::size_t stride1, std::size_t stride2, block_size
){
	static std::size_t const MR = block_size::mr;
	static std::size_t const NR = block_size::nr;
	std::size_t const mp  = (mc+MR-1) / MR;
	std::size_t const np  = (nc+NR-1) / NR;

	for (std::size_t j=0; j<np; ++j) {
		std::size_t const nr = std::min(NR, nc - j*NR);

		for (std::size_t i=0; i<mp; ++i) {
			std::size_t const mr = std::min(MR, mc - i*MR);
			auto CBlockStart = C+i*MR*stride1+j*NR*stride2;
			if (mr==MR && nr==NR) {
				ugemm<block_size>(
					kc, alpha,
					&A[i*kc*MR], &B[j*kc*NR],
					CBlockStart, stride1, stride2
				);
			} else {
				TC CTempBlock[MR*NR];
				std::fill_n(CTempBlock, MR*NR, T(0));
				ugemm<block_size>(
					kc, alpha,
					&A[i*kc*MR], &B[j*kc*NR],
					CTempBlock, NR, 1
				);

				for (std::size_t i0=0; i0<mr; ++i0){
					for (std::size_t j0=0; j0<nr; ++j0) {
						CBlockStart[i0*stride1+j0 * stride2] += CTempBlock[i0*NR+j0];
					}
				}
			}
		}
	}
}


//-- Packing blocks ------------------------------------------------------------
template <class E, class T, class block_size>
void pack_A_dense(matrix_expression<E, cpu_tag> const& A, T* p, block_size)
{
	BOOST_ALIGN_ASSUME_ALIGNED(p, block_size::block::align);

	std::size_t const mc = A().size1();
	std::size_t const kc = A().size2();
	static std::size_t const MR = block_size::mr;
	const std::size_t mp = (mc+MR-1) / MR;

	std::size_t nu = 0;
	for (std::size_t l=0; l<mp; ++l) {
		for (std::size_t j=0; j<kc; ++j) {
			for (std::size_t i = l*MR; i < l*MR + MR; ++i,++nu) {
				p[nu] = (i<mc) ? A()(i,j) : T(0);
			}
		}
	}
}


template <class E, class T, class block_size>
void pack_B_dense(matrix_expression<E, cpu_tag> const& B, T* p, block_size)
{
    BOOST_ALIGN_ASSUME_ALIGNED(p, block_size::block::align);

    std::size_t const kc = B ().size1();
    std::size_t const nc = B ().size2();
    static std::size_t const NR = block_size::nr;
    std::size_t const np = (nc+NR-1) / NR;

	std::size_t nu = 0;
        for (std::size_t l=0; l<np; ++l) {
		for (std::size_t i=0; i<kc; ++i) {
			for (std::size_t j = l*NR; j < l*NR + NR; ++j,++nu){
				p[nu] = (j<nc) ? B()(i,j) : T(0);
			}
		}
        }
}

}}

#endif
