/*!
 *
 *
 * \brief       Dispatches the GETRF algorithm
 *
 * \author      O. Krause
 * \date        2016
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

#ifndef REMORA_KERNELS_GETRF_HPP
#define REMORA_KERNELS_GETRF_HPP


#include "default/getrf.hpp"

namespace remora{namespace kernels {

///\brief Implements the GEneral TRiangular matrix Factorisation GETRF.
///
/// It is better known as the LU decomposition with partial row-pivoting for dense matrices.
/// The algorithm works in place and does not require additional memory.
///
/// The algorithm computes
/// A = P * L * U
///
/// where L is lower unit-triangular and U upper triangular.
/// 
/// The unit diagonal part of L is not stored explicitely. P is a permutation matrix
/// where P(i) stores the index of the row that row i is swapped with.
template <typename MatA, typename VecP>
void getrf(
	matrix_expression<MatA, cpu_tag>& A,
	vector_expression<VecP, cpu_tag>& P
) {
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(P().size() == A().size1());
	return bindings::getrf(A,P);
}

}}
#endif
