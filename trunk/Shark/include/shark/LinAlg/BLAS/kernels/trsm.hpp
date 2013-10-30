/*!
 *  \author O. Krause
 *  \date 2012
 *
 *  \par Copyright (c) 1998-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
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
