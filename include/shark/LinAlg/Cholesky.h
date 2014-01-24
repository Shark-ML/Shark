/*!
 * 
 * \file        Cholesky.h
 *
 * \brief       Cholesky Decompositions for a Matrix A = LL^T
 * 
 * 
 * 
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

#ifndef SHARK_LINALG_CHOLESKY_H
#define SHARK_LINALG_CHOLESKY_H

#include <shark/LinAlg/Base.h>

namespace shark{ namespace blas{

/**
 * \ingroup shark_globals
 * 
 * @{
 */

/*!
 *  \brief Lower triangular Cholesky decomposition.
 *
 *  Given an \f$ m \times m \f$ symmetric positive definite matrix
 *  \f$A\f$, compute the lower triangular matrix \f$L\f$ such that
 *  \f$A = LL^T \f$.
 *  An exception is thrown if the matrix is not positive definite. 
 *  If you suspect the matrix to be positive semi-definite, use
 *  pivotingCholeskyDecomposition instead
 *
 *  \param  A \f$ m \times m \f$ matrix, which must be symmetric and positive definite
 *  \param	L \f$ m \times m \f$ matrix, which stores the Cholesky factor
 *  \return none
 *
 *
 */
template<class MatrixT,class MatrixL>
void choleskyDecomposition(
	matrix_expression<MatrixT> const& A, 
	matrix_expression<MatrixL>& L
);

/*!
 *  \brief Lower triangular Cholesky decomposition with full pivoting performed in place.
 *
 *  Given an \f$ m \times m \f$ symmetric positive semi-definite matrix
 *  \f$A\f$, compute the lower triangular matrix \f$L\f$ and permutation Matrix P such that
 *  \f$P^TAP = LL^T \f$. If matrix A has rank(A) = k, the first k columns of A hold the full
 *  decomposition, while the rest of the matrix is zero. 
 *  This method is slower than the cholesky decomposition without pivoting but numerically more
 *  stable. The diagonal elements are ordered such that i > j => L(i,i) >= L(j,j)
 *
 *  The implementation used here is described in the working paper 
 *  "LAPACK-Style Codes for Level 2 and 3 Pivoted Cholesky Factorizations"
 *  http://www.netlib.org/lapack/lawnspdf/lawn161.pdf
 *
 * The computation is carried out in place this means A is destroied and replaced by L.
 *  
 *
 *  \param  Lref \f$ m \times m \f$ matrix, which must be symmetric and positive definite. It is replaced by L in the end.
 *  \param  P The pivoting matrix
 *  \return The rank of the matrix A
 */
template<class MatrixL>
std::size_t pivotingCholeskyDecompositionInPlace(
	shark::blas::matrix_expression<MatrixL>& Lref,
	PermutationMatrix& P
);

/*!
 *  \brief Lower triangular Cholesky decomposition with full pivoting
 *
 *  Given an \f$ m \times m \f$ symmetric positive semi-definite matrix
 *  \f$A\f$, compute the lower triangular matrix \f$L\f$ and permutation Matrix P such that
 *  \f$P^TAP = LL^T \f$. If matrix A has rank(A) = k, the first k columns of A hold the full
 *  decomposition, while the rest of the matrix is zero. 
 *  This method is slower than the cholesky decomposition without pivoting but numerically more
 *  stable. The diagonal elements are ordered such that i > j => L(i,i) >= L(j,j)
 *
 *  The implementation used here is described in the working paper 
 *  "LAPACK-Style Codes for Level 2 and 3 Pivoted Cholesky Factorizations"
 *  http://www.netlib.org/lapack/lawnspdf/lawn161.pdf
 *  
 *
 *  \param  A \f$ m \times m \f$ matrix, which must be symmetric and positive definite
 *  \param  P The pivoting matrix
 *  \param  L \f$ m \times m \f$ matrix, which stores the Cholesky factor
 *  \return The rank of the matrix A
 *
 *
 */
template<class MatrixA,class MatrixL>
std::size_t pivotingCholeskyDecomposition(
	matrix_expression<MatrixA> const& A,
	PermutationMatrix& P,
	matrix_expression<MatrixL>& L
){	
	//ensure sizes are correct
	SIZE_CHECK(A().size1() == A().size2());
	size_t m = A().size1();
	ensure_size(L,m,m);
	noalias(L()) = A;
	return pivotingCholeskyDecompositionInPlace(L,P);
}

/** @}*/
}}

//implementation of the template functions
#include "Impl/Cholesky.inl"

#endif 
