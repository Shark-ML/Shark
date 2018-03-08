/*!
 * 
 *
 * \brief       matrix-matrix multiplication kernel for symmetrik Rank-K updates
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

#ifndef REMORA_KERNELS_SYRK_HPP
#define REMORA_KERNELS_SYRK_HPP

#include "default/syrk.hpp"

#ifdef REMORA_USE_CBLAS
#include "cblas/syrk.hpp"
#else
//if no bindings are included, we have to provide the default has_optimized_syrk otherwise the binding will take care of this
namespace remora{ namespace bindings{
template<class M1, class M2>
struct  has_optimized_syrk
: public std::false_type{};
}}
#endif

namespace remora{namespace kernels{
	
///\brief Well known SYmmetric Rank-K update kernel M+=alpha*A*A^T.
///
/// Note that it assumes M to be symmetric and it will only touch the upper or lower triangular area.
/// If bindings are included and the matrix combination allow for a specific binding
/// to be applied, the binding is called automatically from {binding}/syrk.h
/// otherwise default/syrk.h is used.
template<bool Upper, class M, class E>
void syrk(
	matrix_expression<E, cpu_tag> const& e,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha
) {
	REMORA_SIZE_CHECK(m().size1() == m().size2());
	REMORA_SIZE_CHECK(m().size1() == e().size1());
	
	bindings::syrk<Upper>(e, m, alpha,
		typename bindings::has_optimized_syrk<M,E>::type()
	);
}

}}

#ifdef REMORA_USE_CLBLAST
#include "clBlast/syrk.hpp"
#elif defined REMORA_USE_GPU
#include "gpu/syrk.hpp"
#endif

#endif
