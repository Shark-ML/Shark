//===========================================================================
/*!
 * 
 *
 * \brief       Functions for solving systems of equations
 * 
 * 
 * 
 *
 * \author      O. Krause
 * \date        2016
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
//===========================================================================

#ifndef SHARK_LINALG_BLAS_SOLVE_H
#define SHARK_LINALG_BLAS_SOLVE_H

#include "kernels/trsm.hpp"
#include "kernels/trsv.hpp"
#include "detail/traits.hpp"
#include "matrix_proxy.hpp"

namespace shark{ namespace blas{

/// \brief In-place triangular linear equation solver.
///
///solves a System of linear equations Ax=b or xA=b
///where A is a lower or upper triangular matrix
///The solution is stored in b afterwards.
///Be aware, that the matrix must have full rank!
///This call needs to template parameters indicating which type of
///system is to be solved : left => Ax=b or right => xA=b
///The second flag indicates which type matrix is used
template<class MatT,class VecT, class Device, bool Upper, bool Unit>
void solve(
	matrix_expression<MatT, Device> const& A, 
	vector_expression<VecT, Device>& b,
	left,
	triangular_tag<Upper, Unit>
){
	kernels::trsv<Upper,Unit>(A,b);
}
/// \brief  In-place triangular linear equation solver.
///
///Solves multiple systems of linear equations
///Ax_1=b_1
///Ax_1=b_2
///...
///=>AX=B or XA=B
///where A is a lower or upper triangular m x m matrix.
///And B = (b_1 b_2 ... b_n) is a m x n matrix.
///The result of X is stored in B afterwards.
///Be aware, that the matrix must have full rank!
///This call needs two template parameters indicating which type of
///system is to be solved : Ax=b or xA=b
///The second flag indicates which type of diagonal is used:
///lower unit, upper unit or non unit lower/upper.
template<class MatA,class MatB, class Device, bool Upper, bool Unit>
void solve(
	matrix_expression<MatA, Device> const& A, 
	matrix_expression<MatB, Device>& B,
	left,
	triangular_tag<Upper, Unit>
){
	kernels::trsm<Upper,Unit>(A,B);
}

/// \brief In-Place solver if A was already cholesky decomposed
///Solves multiple systems of linear equations
///Ax_1=b_1
///Ax_1=b_2
///...
///=>AX=B or XA=B
///given an A which was already Cholesky-decomposed as
///A=LL^T or A=U^TU where L is a lower and U an upper triangular matrix.
template<class System,class MatA,class MatB, class Device>
void solve(
	matrix_expression<MatA, Device> const& A, 
	matrix_expression<MatB, Device>& B,
	left,
	cholesky_tag<lower>
){
	solve(A,B,left(), lower());
	solve(trans(A),B,left(), upper());
}

/// \brief In-Place solver if A was already cholesky decomposed
///Solves multiple systems of linear equations
///Ax_1=b_1
///Ax_1=b_2
///...
///=>AX=B or XA=B
///given an A which was already Cholesky-decomposed as
///A=LL^T or A=U^TU where L is a lower and U an upper triangular matrix.
template<class System,class MatA,class MatB, class Device>
void solve(
	matrix_expression<MatA, Device> const& A, 
	matrix_expression<MatB, Device>& B,
	left,
	cholesky_tag<upper>
){
	solve(trans(A),B,left(), lower());
	solve(A,B,left(), upper());
}

/// \brief In-Place solver if A was already cholesky decomposed
///Solves system of linear equations
///Ax=b
///given an A which was already Cholesky-decomposed as
///A=LL^T or A=U^TU where L is a lower and U an upper triangular matrix.
template<class System,class MatA,class VecT, class Device>
void solve(
	matrix_expression<MatA, Device> const& A, 
	vector_expression<VecT, Device>& b,
	left,
	cholesky_tag<lower>
){
	solve(A,b,left(), lower());
	solve(trans(A),b,left(), upper());
}

/// \brief In-Place solver if A was already cholesky decomposed
///Solves system of linear equations
///Ax=b
///given an A which was already Cholesky-decomposed as
///A=LL^T or A=U^TU where L is a lower and U an upper triangular matrix.
template<class System,class MatA,class VecT, class Device>
void solve(
	matrix_expression<MatA, Device> const& A, 
	vector_expression<VecT, Device>& b,
	left,
	cholesky_tag<upper>
){
	solve(trans(A),b,left(), lower());
	solve(A,b,left(), upper());
}


//final special cases for right systems

template<class MatT,class VecT, class Device, class MatrixType>
void solve(
	matrix_expression<MatT, Device> const& A, 
	vector_expression<VecT, Device>& b,
	right,
	MatrixType
){
	auto transA = trans(A);
	solve(transA,b, left(), typename MatrixType::transposed_orientation());
}

template<class MatA,class MatB, class Device, class MatrixType>
void solve(
	matrix_expression<MatA, Device> const& A, 
	matrix_expression<MatB, Device>& B,
	right,
	MatrixType
){
	auto transA = trans(A);
	auto transB = trans(B);
	solve(transA,transB, left(), typename MatrixType::transposed_orientation());
}


}}
#endif
