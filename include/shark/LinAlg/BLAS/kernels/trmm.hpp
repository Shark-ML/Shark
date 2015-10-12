/*!
 * 
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

#ifndef SHARK_LINALG_BLAS_KERNELS_TRMM_HPP
#define SHARK_LINALG_BLAS_KERNELS_TRMM_HPP

#ifdef SHARK_USE_CBLAS
#include "atlas/trmm.hpp"
#else
// if no bindings are included, we have to provide the default has_optimized_gemv 
// otherwise the binding will take care of this
namespace shark { namespace blas { namespace bindings{
template<class M1, class M2>
struct  has_optimized_trmm
: public boost::mpl::false_{};
}}}
#endif

#include "default/trmm.hpp"

namespace shark { namespace blas {namespace kernels{
	
///\brief Implements the TRiangular Matrix Matrix multiply.
///
/// It computes B=A*B in place, where A is a triangular matrix and B a dense matrix
template <bool Upper,bool Unit,typename TriangularA, typename MatB>
void trmm(
	matrix_expression<TriangularA> const &A, 
	matrix_expression<MatB>& B
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size1() == B().size1());
	
	bindings::trmm<Upper,Unit>(A,B,typename bindings::has_optimized_trmm<TriangularA, MatB>::type());
}

}}}

#endif
