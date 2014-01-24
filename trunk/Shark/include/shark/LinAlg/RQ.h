/*!
 * 
 * \file        RQ.h
 *
 * \brief       Algorithm for Triangular-Quadratic-Decomposition
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
#ifndef SHARK_LINALG_RQ_H
#define SHARK_LINALG_RQ_H

#include <shark/LinAlg/Base.h>
namespace shark{ namespace blas{

/**
 * \ingroup shark_globals
 * 
 * @{
 */


	
/*!
 *  \brief Determines the RQ Decomposition of the matrix A using pivoting 
 *   returning the housholder transformation instead of Q.
 *
 * The pivoting RQ-Decomposition finds an orthonormal matrix Q and a lower Triangular matrix R
 * as well as a permuation matrix P such that PA = R*Q. 
 * Since Q is the multiplication of all householder transformations,
 * It is quite expensive to compute. Often, Q is only an intermediate step in computations which can be
 * carried out more efficiently using the Householder Transformations themselves.
 *
 * The Matrix format of the householder transform is that the transformations are stored as 
 * upper triangular matrix. The first transformation being in the first row and so on.
 */
template<class MatrixT,class Mat>
std::size_t pivotingRQ
(
	blas::matrix_expression<MatrixT> const& matrixA,
	blas::matrix_container<Mat>& matrixR,
	blas::matrix_container<Mat>& matrixQ,
	blas::permutation_matrix& permutation
);

/*!
 *  \brief Determines the RQ Decomposition of the matrix A using pivoting
 *
 * The pivoting RQ-Decomposition finds an orthonormal matrix Q and a lower Triangular matrix R
 * as well as a permuation matrix P such that PA = R*Q. 
 * This function is better known as the QR-Decomposition
 * of a transposed matrix B^T = A and B = QR. We depart from the well known algorithm
 * because it is intended to be used with column major matrices. But since shark uses
 * row-major, a QR decomposition is a lot slower. 
 *
 * This Version of the algorithm is based on householder transformations. since it uses pivoting it can
 * be used to determine the rank of a matrix.
 */
template<class MatrixT,class MatrixU>
std::size_t pivotingRQHouseholder
(
	blas::matrix_expression<MatrixT> const& matrixA,
	blas::matrix_container<MatrixU>& matrixR,
	blas::matrix_container<MatrixU>& householderTransform,
	blas::permutation_matrix& permutation
);


/** @}*/
}}

#include "Impl/pivotingRQ.inl"
#endif
