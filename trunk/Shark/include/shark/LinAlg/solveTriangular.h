//===========================================================================
/*!
 *  \brief Some operations for matrices.
 *
 *
 *  \author  O. Krause
 *  \date    2011
 *
 *  \par Copyright (c) 1999-2011:
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

///\brief Flag indicating that the matrix is Upper triangular
struct Upper{
	static const bool upper = true;
	static const bool unit = false;
};
///\brief Flag indicating that the matrix is Upper triangular and diagonal elements are to be assumed as 1
struct UnitUpper{
	static const bool upper = true;
	static const bool unit = true;
};
///\brief Flag indicating that the matrix is Lower triangular
struct Lower{
	static const bool upper = false;
	static const bool unit = false;
};
///\brief Flag indicating that the matrix is Lower triangular and diagonal elements are to be assumed as 1
struct UnitLower{
	static const bool upper = false;
	static const bool unit = true;
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
template<class System,class MatA,class MatB>
void solveTriangularCholeskyInPlace(
	const matrix_expression<MatA>& A, 
	matrix_expression<MatB>& B
);

/// \brief In-Place solver if A was already cholesky decomposed
///Solves system of linear equations
///Ax=b
///given an A which was already Cholesky-decomposed as
///A=LL^T where L is a lower triangular matrix.
template<class System,class MatA,class MatB>
void solveTriangularCholeskyInPlace(
	const matrix_expression<MatA>& A, 
	vector_expression<MatB>& B
);

}}
#include "Impl/solveTriangular.inl"
#endif
