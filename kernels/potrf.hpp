/*!
 *
 *
 * \brief       Dispatches the POTRF algorithm
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

#ifndef REMORA_KERNELS_POTRF_HPP
#define REMORA_KERNELS_POTRF_HPP

#ifdef REMORA_USE_ATLAS_LAPACK
#include "atlas/potrf.hpp"
#else
// if no bindings are included, we have to provide the default has_optimized_gemv
// otherwise the binding will take care of this
namespace remora {namespace bindings {
template<class M>
struct  has_optimized_potrf
	: public std::false_type {};
}}
#endif

#include "default/potrf.hpp"

namespace remora {namespace kernels {

///\brief Implements the POsitive TRiangular matrix Factorisation POTRF.
///
/// It is better known as the cholesky decomposition for dense matrices.
/// The algorithm works in place and does not require additional memory.
template <class Triangular, typename MatA>
std::size_t potrf(
    matrix_container<MatA, cpu_tag>& A
) {
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	return bindings::potrf<Triangular>(A, typename bindings::has_optimized_potrf<MatA>::type());
}

}}

#ifdef REMORA_USE_GPU
#include "gpu/potrf.hpp"
#endif
#endif
