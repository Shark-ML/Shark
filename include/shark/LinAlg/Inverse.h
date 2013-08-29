/*!
 *  \brief Matrix inverses.
 *
 *
 *  \author  O. Krause
 *  \date    2010
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

#ifndef SHARK_LINALG_INVERSE_H
#define SHARK_LINALG_INVERSE_H

#include <shark/LinAlg/Cholesky.h>
namespace shark{ namespace blas{

/**
 * \ingroup shark_globals
 * 
 * @{
 */


//! \brief Inverts a matrix with full rank.
template<class MatrixT>
RealMatrix invert(const MatrixT& mat);

// //! Inverts a symmetric matrix
//template<class MatrixT,class MatrixU>
//void invertSymm(MatrixT &I, const MatrixU& A);

//! Inverts a symmetric positive definite matrix
template<class MatrixT,class MatrixU>
void invertSymmPositiveDefinite(MatrixT &I, const MatrixU& ArrSymm);

/// \brief For a given square matrix A computes a matrix U where A'=LU^T
///
/// A' is the generalized inverse of A. If A has full rank, the resulting matrix
/// is equivalent to the inverse cholesky decomposition. If it is not, it is not necessarily 
/// triangular. This is mostly a helperfunction for g_inverse but also used in
/// other parts of shark
template<class MatA, class MatU>
void decomposedGeneralInverse(
	matrix_expression<MatA> const& matA,
	matrix_expression<MatU>& matU
);

/*!
 *  \brief Calculates the generalized inverse matrix of input matrix "matrixA".
 *
 *  Given an \f$ m \times n \f$ input matrix \f$ A \f$ this function uses 
 *  a Pivoting cholesky decomposition to compute the generalized Inverse with the 
 *  property \f$ AA'A = A \f$
 *
 *  if m < n the matrix also satisfies
 *  AA' = I
 *
 *  If m > n the least squares solution is used.
 *
 *      \param  matrixA \f$ m \times n \f$ input matrix.
 *      \returns The generalised inverse matrix of A.
 */
template<class MatrixT>
RealMatrix g_inverse(matrix_expression<MatrixT> const& matrixA);

/** @}*/
}}

//implementation of the template functions
#include "Impl/invert.inl"
#include "Impl/g_inverse.inl"

#endif 
