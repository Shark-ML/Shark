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

#ifndef REMORA_KERNELS_PSTRF_HPP
#define REMORA_KERNELS_PSTRF_HPP

#include "default/pstrf.hpp"

namespace remora {
namespace kernels {

/*!
 *  \brief Cholesky decomposition with full pivoting performed in place.
 *
 *  Given an \f$ m \times m \f$ symmetric positive semi-definite matrix
 *  \f$A\f$, compute thes matrix \f$L\f$ and permutation Matrix P such that
 *  \f$P^TAP = LL^T \f$. If matrix A has rank(A) = k, the first k columns of A hold the full
 *  decomposition, while the rest of the matrix is zero. 
 *  This method is slower than the cholesky decomposition without pivoting but numerically more
 *  stable. The diagonal elements are ordered such that i > j => L(i,i) >= L(j,j)
 *
 *  The implementation used here is described in the working paper 
 *  "LAPACK-Style Codes for Level 2 and 3 Pivoted Cholesky Factorizations"
 *  http://www.netlib.org/lapack/lawnspdf/lawn161.pdf
 *
 * The computation is carried out in place this means A is destroyed and replaced by L.
 *  
 *
 *  \param  A \f$ m \times m \f$ matrix, which must be symmetric and positive definite. It is replaced by L in the end.
 *  \param  P The pivoting matrix of dimension \f$ m \f$
 *  \return The rank of the matrix A
 */
template<class Triangular, class MatA, class VecP>
std::size_t pstrf(
	matrix_expression<MatA, cpu_tag>&A,
	vector_expression<VecP, cpu_tag>& P
){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(P().size() == A().size1());
	return bindings::pstrf(A,P, Triangular());
}


}}

#endif
