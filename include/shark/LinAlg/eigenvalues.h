/*!
 * 
 *
 * \brief       Algorithms for Eigenvalue decompositions
 * 
 * 
 * 
 *
 * \author      O. Krause
 * \date        2011
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
#ifndef SHARK_LINALG_EIGENVALUES_H
#define SHARK_LINALG_EIGENVALUES_H

#include <shark/LinAlg/Base.h>
#include <shark/LinAlg/BLAS/kernels/syev.hpp>

namespace shark{ namespace blas{

/**
 * \ingroup shark_globals
 * 
 * @{
 */

/*!
 *  \brief Used as frontend for
 *  eigensymm for calculating the eigenvalues and the normalized eigenvectors of a symmetric matrix
 *  'A' using the Givens and Householder reduction. Each time this frontend is called additional
 *  memory is allocated for intermediate results.
 *
 *
 * \param A \f$ n \times n \f$ matrix, which must be symmetric, so only the bottom triangular matrix must contain values.
 * \param eigenVectors \f$ n \times n \f$ matrix with the calculated normalized eigenvectors, each column contains an eigenvector.
 * \param eigenValues n-dimensional vector with the calculated eigenvalues in descending order.
 * \return none.
 *
 * \throw SharkException
 */
template<class MatrixT,class MatrixU,class VectorT>
void eigensymm
(
	matrix_expression<MatrixT> const& A,
	matrix_expression<MatrixU>& eigenVectors,
	vector_expression<VectorT>& eigenValues
)
{
	SIZE_CHECK(A().size2() == A().size1());
	std::size_t n = A().size1();
	
	eigenVectors().resize(n,n);
	eigenVectors().clear();
	eigenValues().resize(n);
	eigenValues().clear();
	// special case n = 1
	if (n == 1) {
		eigenVectors()( 0 ,  0 ) = 1;
		eigenValues()( 0 ) = A()( 0 ,  0 );
		return;
	}

	// copy matrix
	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j <= i; j++) {
			eigenVectors()(i, j) = A()(i, j);
		}
	}
	
	kernels::syev(eigenVectors,eigenValues);
}



/** @}*/
}}
#endif
