/*!
 * 
 *
 * \brief       matrix-matrix multiplication kernel for symmetric matrices
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

#ifndef SHARK_LINALG_BLAS_KERNELS_SYMM_HPP
#define SHARK_LINALG_BLAS_KERNELS_SYMM_HPP

#include "gemm.hpp"

namespace shark { namespace blas {namespace kernels{
	
///\brief Well known SYmmetric Matrix-Matrix product kernel M+=alpha*E*E^T.
///
/// This kernel uses the fact that symmetric matrices can be stored as triangular matrix,
/// therefore only the lower or upper part of the matrix is consideres, saving 50% of FLOPS
template<class Triangular, class M, class E>
void symm(
	matrix_expression<E1, cpu_tag> const& e1,
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type alpha
) {
	SIZE_CHECK(m().size1() == e1().size1());
	SIZE_CHECK(m().size1() == m().size2());
	
	gemm(e1, trans(e1), m,alpha);
}

}}}
#endif
