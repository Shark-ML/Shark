//===========================================================================
/*!
 * 
 *
 * \brief       Some operations for matrices.
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
//===========================================================================

#ifndef SHARK_LINALG_SOLVE_TRIANGULAR_SYSTEM_H
#define SHARK_LINALG_SOLVE_TRIANGULAR_SYSTEM_H

#include <shark/LinAlg/Base.h>

namespace shark{ namespace blas{

//a few flags governing which type of system is to be solved

///\brief Flag indicating that a system AX=B is to be solved
struct SolveAXB{
	static const bool left = true;
};
///\brief Flag indicating that a system XA=B is to be solved
struct SolveXAB{
	static const bool left = false;
};

/// \brief In-place triangular linear equation solver.
///
///solves a System of linear equations Ax=b or xA=b
///where A is a lower or upper triangular matrix
///The solution is stored in b afterwards.
///Be aware, that the matrix must have full rank!
///This call needs to template parameters indicating which type of
///system is to be solved : Ax=b or xA=b
///The second flag indicates which type of diagonal is used:
///lower unit, upper unit or non unit lower/upper.
template<class System, class DiagType,class MatT,class VecT>
void solveTriangularSystemInPlace(
	const matrix_expression<MatT>& A, 
	vector_expression<VecT>& b
);
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
template<class System, class DiagType,class MatA,class MatB>
void solveTriangularSystemInPlace(
	const matrix_expression<MatA>& A, 
	matrix_expression<MatB>& B
);

/// \brief In-Place solver if A was already cholesky decomposed
///Solves multiple systems of linear equations
///Ax_1=b_1
///Ax_1=b_2
///...
///=>AX=B or XA=B
///given an A which was already Cholesky-decomposed as
///A=LL^T where L is a lower triangular matrix.
template<class System,class MatL,class MatB>
void solveTriangularCholeskyInPlace(
	const matrix_expression<MatL>&L, 
	matrix_expression<MatB>& B
);

/// \brief In-Place solver if A was already cholesky decomposed
///Solves system of linear equations
///Ax=b
///given an A which was already Cholesky-decomposed as
///A=LL^T where L is a lower triangular matrix.
template<class System,class MatL,class VecB>
void solveTriangularCholeskyInPlace(
	const matrix_expression<MatL>& L, 
	vector_expression<VecB>& b
);

}}
#include "Impl/solveTriangular.inl"
#endif
