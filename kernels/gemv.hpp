/*!
 * 
 *
 * \brief       matrix-vector multiplication kernel
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
#ifndef REMORA_KERNELS_GEMV_HPP
#define REMORA_KERNELS_GEMV_HPP

#include "default/gemv.hpp"

#ifdef REMORA_USE_CBLAS
#include "cblas/gemv.hpp"
#else
// if no bindings are included, we have to provide the default has_optimized_gemv 
// otherwise the binding will take care of this
namespace remora{ namespace bindings{
template<class M1, class M2, class M3>
struct  has_optimized_gemv
: public std::false_type{};
}}
#endif

#include <cassert>
	
namespace remora{namespace kernels{
	
///\brief Well known GEneral Matrix-Vector product kernel M+=alpha*E1*e2.
///
/// If bindings are included and the matrix/vector combination allows for a specific binding
/// to be applied, the binding is called automatically from {binding}/gemv.h
/// otherwise default/gemv.h is used which is fully implemented for all dense/sparse combinations.
/// if a combination is optimized, bindings::has_optimized_gemv<M,E1,E2>::type evaluates to std::true_type
/// The kernels themselves are implemented in bindings::gemv.
template<class M, class E1, class E2>
void gemv(
	matrix_expression<E1, cpu_tag> const& e1,
	vector_expression<E2, cpu_tag> const& e2,
	vector_expression<M, cpu_tag>& m,
	typename M::value_type alpha
) {
	assert(m().size() == e1().size1());
	assert(e1().size2() == e2().size());
	
	bindings::gemv(
		e1, e2, m,alpha,
		typename bindings::has_optimized_gemv<M,E1,E2>::type()
	);
}

}}

#ifdef REMORA_USE_CLBLAST
#include "clBlast/gemv.hpp"
#elif defined REMORA_USE_GPU
#include "gpu/gemv.hpp"
#endif

#endif