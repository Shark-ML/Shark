/*!
 * 
 * \file        trsm.hpp
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

#ifndef SHARK_LINALG_BLAS_UBLAS_KERNELS_TRSM_HPP
#define SHARK_LINALG_BLAS_UBLAS_KERNELS_TRSM_HPP

#ifdef SHARK_USE_ATLAS
#include "atlas/trsm.hpp"
#else
// if no bindings are included, we have to provide the default has_optimized_gemv 
// otherwise the binding will take care of this
namespace shark { namespace blas { namespace bindings{
template<class M1, class M2>
struct  has_optimized_trsm
: public boost::mpl::false_{};
}}}
#endif

#include "default/trsm.hpp"

namespace shark { namespace blas {namespace kernels{
	
///\brief Implements the TRiangular Solver for Vectors.
///
/// It solves Systems of the form Ax = b where A is a square lower or upper triangular matrix.
/// It can optionally assume that the diagonal is 1 and won't access the diagonal elements.
template <bool Upper,bool Unit,typename TriangularA, typename MatB>
void trsm(
	matrix_expression<TriangularA> const &A, 
	matrix_expression<MatB> &B
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size1() == B().size1());
	
	bindings::trsm<Upper,Unit>(A,B,typename bindings::has_optimized_trsm<TriangularA, MatB>::type());
}

}}}

#endif
