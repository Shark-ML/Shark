/*!
 *
 *
 * \brief       matrix-matrix multiplication kernel
 *
 * \author      O. Krause
 * \date        2012
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

#ifndef REMORA_KERNELS_GEMM_HPP
#define REMORA_KERNELS_GEMM_HPP

#include "default/gemm.hpp"
#ifdef REMORA_USE_CBLAS
#include "cblas/dense_gemm.hpp"
#else
#include "default/dense_gemm.hpp"
#endif

#include "../proxy_expressions.hpp"

namespace remora{

namespace bindings{
	//-- Dense gemm
	template <class E1, class E2, class Mat, class Orientation1, class Orientation2>
	void gemm(
		matrix_expression<E1, cpu_tag> const& e1,
		matrix_expression<E2, cpu_tag> const& e2,
		matrix_expression<Mat, cpu_tag>& m,
		typename Mat::value_type alpha,
		row_major, Orientation1, Orientation2,
		dense_tag, dense_tag
	){
		dense_gemm(e1,e2,m,alpha);
	}
	//column major result is transformed to row_major using A=B*C <=> A^T = C^T B^T
	template<class M, class E1, class E2, class Orientation1, class Orientation2, class Tag1, class Tag2>
	void gemm(
		matrix_expression<E1, cpu_tag> const& e1,
		matrix_expression<E2, cpu_tag> const& e2,
		matrix_expression<M, cpu_tag>& m,
		typename M::value_type alpha,
		column_major, Orientation1, Orientation2,
		Tag1, Tag2
	){
		auto transposedM = trans(m);
		typedef typename Orientation1::transposed_orientation transpO1;
		typedef typename Orientation2::transposed_orientation transpO2;
		gemm(trans(e2),trans(e1),transposedM,alpha,row_major(),transpO2(),transpO1(), Tag2(),Tag1());
	}
}


namespace kernels{

///\brief Well known GEneral Matrix-Matrix product kernel M+=alpha*E1*E2.
///
/// If bindings are included and the matrix combination allow for a specific binding
/// to be applied, the binding is called automatically from {binding}/gemm.h
/// otherwise default/gemm.h is used which is fully implemented for all dense/sparse combinations.
/// if a combination is optimized, bindings::has_optimized_gemm<M,E1,E2>::type evaluates to std::true_type
/// The kernels themselves are implemented in bindings::gemm.
template<class M, class E1, class E2>
void gemm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<E2, cpu_tag> const& e2,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha
) {
	REMORA_SIZE_CHECK(m().size1() == e1().size1());
	REMORA_SIZE_CHECK(m().size2() == e2().size2());
	REMORA_SIZE_CHECK(e1().size2() == e2().size1());

	typedef typename M::orientation ResultOrientation;
	typedef typename E1::orientation E1Orientation;
	typedef typename E2::orientation E2Orientation;
	typedef typename E1::evaluation_category::tag E1Tag;
	typedef typename E2::evaluation_category::tag E2Tag;

	bindings::gemm(e1, e2, m ,alpha,
		ResultOrientation(), E1Orientation(), E2Orientation(),
		E1Tag(),E2Tag()
	);
}

}}

#ifdef REMORA_USE_CLBLAST
#include "clBlast/gemm.hpp"
#elif defined REMORA_USE_GPU
#include "gpu/gemm.hpp"
#endif
#endif
