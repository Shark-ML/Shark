/*!
 * 
 *
 * \brief       Triangular matrix-vector multiplication kernel
 *
 * \author      O. Krause
 * \date        2012
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

#ifndef SHARK_LINALG_BLAS_KERNELS_TRMV_HPP
#define SHARK_LINALG_BLAS_KERNELS_TRMV_HPP

#ifdef SHARK_USE_CBLAS
#include "cblas/trmv.hpp"
#else
// if no bindings are included, we have to provide the default has_optimized_gemv 
// otherwise the binding will take care of this
namespace shark { namespace blas { namespace bindings{
template<class M, class V>
struct  has_optimized_trmv
: public boost::mpl::false_{};
}}}
#endif

#include "default/trmv.hpp"

namespace shark { namespace blas {namespace kernels{
	
///\brief Implements the TRiangular Solver for Vectors.
///
/// It solves Systems of the form Ax = b where A is a square lower or upper triangular matrix.
/// It can optionally assume that the diagonal is 1 and won't access the diagonal elements.
template <bool Upper,bool Unit,typename MatA, typename VecB>
void trmv(
	matrix_expression<MatA, cpu_tag> const &A, 
	vector_expression<VecB, cpu_tag>& b
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size1() == b().size());
	
	bindings::trmv<Upper,Unit>(A,b,typename bindings::has_optimized_trmv<MatA, VecB>::type());
}

}}}

#ifdef SHARK_USE_CLBLAS
#include "clblas/trmv.hpp"
#endif

#endif
