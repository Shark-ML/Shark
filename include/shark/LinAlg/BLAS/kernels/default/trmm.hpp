/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2010
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

#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_TRMM_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_TRMM_HPP

#include <shark/LinAlg/BLAS/kernels/trmv.hpp>
#include "../../matrix_proxy.hpp"

namespace shark { namespace blas { namespace bindings {

template <bool Upper,bool Unit,typename TriangularA, typename MatB>
void trmm(
	matrix_expression<TriangularA> const &A, 
	matrix_expression<MatB>& B,
	boost::mpl::false_
) {
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size1() == B().size1());
	
	std::size_t numCols=B().size2();
	
	for(std::size_t  i = 0; i != numCols; ++i){
		matrix_column<MatB> col = column(B,i);
		kernels::trmv<Upper,Unit>(A,col);
	}
}

}}}

#endif
