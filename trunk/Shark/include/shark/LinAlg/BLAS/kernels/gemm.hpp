/*!
 * 
 * \file        gemm.hpp
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2012
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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

#ifndef SHARK_LINALG_BLAS_UBLAS_KERNELS_GEMM_HPP
#define SHARK_LINALG_BLAS_UBLAS_KERNELS_GEMM_HPP

#include "default/gemm.hpp"

#ifdef SHARK_USE_ATLAS
#include "atlas/gemm.hpp"
#else
//if no bindings are included, we have to provide the default has_optimized_gemm otherwise the binding will take care of this
namespace shark { namespace blas { namespace bindings{
template<class M1, class M2, class M3>
struct  has_optimized_gemm
: public boost::mpl::false_{};
}}}
#endif
	
namespace shark { namespace blas {namespace kernels{
	
///\brief Well known GEneral Matrix-Matrix product kernel M+=alpha*E1*E2.
///
/// If bindings are included and the matrix combination allow for a specific binding
/// to be applied, the binding is called automatically from {binding}/gemm.h
/// otherwise default/gemm.h is used which is fully implemented for all dense/sparse combinations.
/// if a combination is optimized, bindings::has_optimized_gemm<M,E1,E2>::type evaluates to boost::mpl::true_
/// The kernels themselves are implemented in blas::bindings::gemm.
template<class M, class E1, class E2>
void gemm(
	matrix_expression<E1> const& e1,
	matrix_expression<E2> const& e2,
	matrix_expression<M>& m,
	typename M::value_type alpha
) {
	SIZE_CHECK(m().size1() == e1().size1());
	SIZE_CHECK(m().size2() == e2().size2());
	SIZE_CHECK(e1().size2() == e2().size1());
	
	bindings::gemm(
		e1, e2, m,alpha,
		typename bindings::has_optimized_gemm<M,E1,E2>::type()
	);
}

}}}

#endif
