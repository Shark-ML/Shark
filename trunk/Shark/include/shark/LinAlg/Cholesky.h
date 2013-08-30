/*!
 *  \brief Cholesky Decompositions for a Matrix A = LL^T
 *
 *
 *  \author  O. Krause
 *  \date    2012
 *
 *  \par Copyright (c) 1999-2001:
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
 *
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
template<class MatrixT,class MatrixL>
std::size_t pivotingCholeskyDecomposition(
	matrix_expression<MatrixT> const& A,
	PermutationMatrix& P,
	matrix_expression<MatrixL>& L
);

/** @}*/
}}

//implementation of the template functions
#include "Impl/Cholesky.inl"

#endif 
