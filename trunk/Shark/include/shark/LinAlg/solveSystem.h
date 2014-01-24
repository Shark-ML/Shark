//===========================================================================
/*!
 * 
 * \file        solveSystem.h
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

#ifndef SHARK_LINALG_SOLVE_SYSTEM_H
#define SHARK_LINALG_SOLVE_SYSTEM_H

#include <shark/LinAlg/solveTriangular.h>

namespace shark{ namespace blas{

/**
 * \ingroup shark_globals
 * 
 * @{
 */
	
/// \brief In-Place System of linear equations solver.
///
///Solves a system of linear equations
///Ax=b 
///for x, using LU decomposition and
///backward substitution sotring the results in b. 
///This Method is in
///no way optimized for sparse matrices.
///Be aware, that the matrix must have full rank!
template<class MatT,class VecT>
void solveSystemInPlace(
	matrix_expression<MatT> const& A, 
	vector_expression<VecT>& b
);
/// \brief System of linear equations solver.
/// 
/// Solves asystem of linear equations
/// Ax=b 
/// for x, using LU decomposition and
/// backward substitution. This Method is in
/// no way optimized for sparse matrices.
/// Be aware, that the matrix must have full rank!
template<class MatT,class Vec1T,class Vec2T>
void solveSystem(
	const matrix_expression<MatT> & A, 
	vector_expression<Vec1T>& x,
	const vector_expression<Vec2T> & b
);

/// \brief In-Place system of linear equations solver.
///
///Solves multiple systems of linear equations
///Ax_1=b_1
///Ax_2=b_2
///...
///=>AX=B
///for X, using LU decomposition and
///backward substitution and stores the result in b
///Note, that B=(b_1,...,b_n), so the right hand sides are stored as columns
///This Method is in no way optimized for sparse matrices.
///Be aware, that the matrix must have full rank!
template<class MatT,class Mat2T>
void solveSystemInPlace(
	matrix_expression<MatT> const& A, 
	matrix_expression<Mat2T>& B
);
/// \brief System of linear equations solver.
/// 
/// Solves multiple systems of linear equations
/// Ax_1=b_1
/// Ax_2=b_2
/// ...
/// =>AX=B
/// for X, using LU decomposition and
/// backward substitution.
/// Note, that B=(b_1,...,b_n), so the right hand sides are stored as columns
/// This Method is in no way optimized for sparse matrices.
/// Be aware that the matrix must have full rank!
template<class MatT,class Mat1T,class Mat2T>
void solveSystem(
	const matrix_expression<MatT> & A, 
	matrix_expression<Mat1T>& X,
	const matrix_expression<Mat2T> & B
);

/// \brief System of symmetric linear equations solver. The result is stored in b
/// 
/// Solves a system of linear equations
/// Ax=b 
/// for x, using Cholesky decomposition and
/// backward substitution. and stores the result in b.
/// A must be symmetric.
/// This method is in no way optimized for sparse matrices.
/// Be aware, that the matrix must have full rank!
template<class System, class MatT,class VecT>
void solveSymmSystemInPlace(
	matrix_expression<MatT> const& A,
	vector_expression<VecT>& b
);

/// \brief System of symmetric linear equations solver.
/// 
/// Solves multiple systems of linear equations
/// Ax_1=b_1
/// Ax_1=b_2
/// ...
/// =>AX=B
/// or XA=B
/// for X, using cholesky decomposition and
/// backward substitution. The first template parameter is used 
/// to decide which type of system is solved
/// Note, that B=(b_1,...,b_n), so the right hand sides are stored as columns.
/// A must be symmetric.
/// This Method is in no way optimized for sparse matrices.
/// Be aware, that the matrix must have full rank!
/// Also the result is stored in B directly so it"s contents are destroyed.
/// @param A the system matrix A
/// @param B the right hand side of the LGS, also stores the result
template<class System, class MatT,class Mat1T>
void solveSymmSystemInPlace(
	matrix_expression<MatT> const& A,
	matrix_expression<Mat1T>& B
);

/// \brief System of symmetric linear equations solver.
/// 
/// Solves a system of linear equations
/// Ax=b 
/// for x, using Cholesky decomposition and
/// backward substitution. A must be symmetric.
/// This Method is in no way optimized for sparse matrices.
/// Be aware, that the matrix must have full rank!
template<class System,class MatT,class Vec1T,class Vec2T>
void solveSymmSystem(
	matrix_expression<MatT> const& A, 
	vector_expression<Vec1T>& x,
	vector_expression<Vec2T> const& b
);
/// \brief System of symmetric linear equations solver.
/// 
/// Solves multiple systems of linear equations
/// Ax_1=b_1
/// Ax_1=b_2
/// ...
/// =>AX=B
/// or XA = B
/// for X, using cholesky decomposition and
/// backward substitution. The first template parameter is used 
/// to decide which type of system is solved
/// Note, that B=(b_1,...,b_n), so the right hand sides are stored as columns.
/// A must be symmetric.
/// This Method is in no way optimized for sparse matrices.
/// Be aware, that the matrix must have full rank!
/// @param A the system matrix A
/// @param X the stored result of the solution of LGS
/// @param B the right hand side of the LGS
template<class System,class MatT,class Mat1T,class Mat2T>
void solveSymmSystem(
	matrix_expression<MatT> const& A, 
	matrix_expression<Mat1T>& X,
	matrix_expression<Mat2T> const& B
);


/// \brief Solves a square system of linear equations without full rank.
/// 
/// Solves the system Ax= b or x^TA=b^T when A is
/// symmetric positive semi-definite.
/// If b is not in the span of Ax or xA, the least squares solution is used,
/// that is we minimize ||Ax-b||^2
///
/// The computation is carried out in-place.
/// The algorithm can be looked up in
/// "Fast Computation of Moore-Penrose Inverse Matrices"
///  Pierre Courrieu, 2005 
/// 
/// \param A \f$ n \times n \f$ input matrix.
/// \param b right hand side vector.
template<class System,class MatT,class VecT>
void solveSymmSemiDefiniteSystemInPlace(
	matrix_expression<MatT> const& A, 
	vector_expression<VecT>& b
);

/// \brief Solves multiple square system of linear equations without full rank.
/// 
/// Solves multiple systems of linear equations
/// Ax_1=b_1
/// Ax_1=b_2
/// ...
/// =>AX=B
/// or XA = B
/// A must be symmetric positive semi-definite - thus is not required to have full rank.
/// Note, that B=(b_1,...,b_n), so the right hand sides are stored as columns.
/// If the b_i are not in the span of Ax_i or x_i^TA, the least squares solution is used,
/// that is we minimize ||Ax_i-b_i||^2
///
/// The computation is carried out in-place.
/// The algorithm can be looked up in
/// "Fast Computation of Moore-Penrose Inverse Matrices"
///  Pierre Courrieu, 2005 
/// 
/// \param A \f$ n \times n \f$ input matrix.
/// \param B \f$ n \times k \f$ right hand side matrix.
template<class System,class Mat1T,class Mat2T>
void solveSymmSemiDefiniteSystemInPlace(
	matrix_expression<Mat1T> const& A, 
	matrix_expression<Mat2T>& B
);

/// \brief Solves a non-square system of linear equations.
/// 
/// Given a \f$ m \times n \f$ input matrix A this function uses 
/// the generalized inverse of A to solve the system of linear equations.
/// If b is not in the span of Ax or xA, the least squares solution is used,
/// that is we minimize ||Ax-b||^2
/// 
/// The computation is carried out in-place.
///
/// \param A \f$ n \times m \f$ input matrix.
/// \param b right hand side of the problem.
template<class System,class MatT,class VecT>
void generalSolveSystemInPlace(
	matrix_expression<MatT> const& A, 
	vector_expression<VecT>& b
);


/// \brief Solves multiple non-square systems of linear equations.
/// 
/// Given a \f$ m \times n \f$ input matrix A this function uses 
/// the generalized inverse of A to solve the system of linear equations AX=B.
/// If b_i is not in the span of Ax_i or x_iA, the least squares solution is used,
/// that is we minimize ||Ax_i-b_i||^2 for all columns b_i of B.
/// 
/// The computation is carried out in-place.
///
/// \param A \f$ n \times m \f$ input matrix.
/// \param B \f$ n \times k \f$ right hand sied matrix.
template<class System,class MatA,class MatB>
void generalSolveSystemInPlace(
	matrix_expression<MatA> const& A, 
	matrix_expression<MatB>& B
);

/// \brief Approximates the solution of a linear system of equation Ax=b.
/// 
/// Most often there is no need for the exact solution of a system of linear
/// equations. Instead only a good approximation needs to be found.
/// In this case an iterative method can be used which stops when
/// a suitable exact solution is found. For a lot of systems this already happens
/// after a very low number of iterations.
/// Every iteration has complexity O(n^2) and after n iterations the
/// exact solution is found. However if this solution is needed, the other
/// methods, as for example solveSymmSystem are more suitable.
/// 
/// This algorithm does not require A to have full rank, however it must be
/// positive semi-definite.
/// 
/// This algorithm stops after the maximum number of iterations is
/// exceeded or after the max-norm of the residual \f$ r_k= Ax_k-b\f$ is 
/// smaller than epsilon.
/// 
/// \param A the positive semi-definite n x n-Matrix
/// \param x the solution vector
/// \param b the right hand side
/// \param epsilon stopping criterium for the residual
/// \param maxIterations the maximum number of iterations
template<class MatT, class VecT>
void approxSolveSymmSystem(
	matrix_expression<MatT> const& A,
	vector_expression<VecT>& x,
	vector_expression<VecT> const& b,
	double epsilon = 1.e-10,
	unsigned int maxIterations = 0
){
	SIZE_CHECK(A().size1()==A().size2());
	SIZE_CHECK(A().size1()==b().size());
	
	std::size_t dim = b().size();
	ensure_size(x,dim);
	x().clear();
	
	unsigned int maxIt = (maxIterations == 0)? dim: maxIterations;
	
	typedef typename VecT::value_type value_type;
	vector<value_type> r=b;
	vector<value_type> rnext(dim);
	vector<value_type> p=b;
	vector<value_type> Ap(dim);
	
	for(std::size_t i = 0; i != maxIt; ++i){
		axpy_prod(A,p,Ap);
		double rsqr=inner_prod(r,r);
		double alpha = rsqr/inner_prod(p,Ap);
		noalias(x())+=alpha*p;
		noalias(rnext) = r - alpha * Ap; 
		if(norm_inf(rnext)<epsilon)
			break;
		
		double beta = inner_prod(rnext,rnext)/rsqr;
		p*=beta;
		noalias(p) +=rnext;
		swap(r,rnext);
	}
}

/// \brief Approximates the solution of a linear system of equation Ax=b, storing the solution in b.
/// 
/// Most often there is no need for the exact solution of a system of linear
/// equations. Instead only a good approximation needs to be found.
/// In this case an iterative method can be used which stops when
/// a suitable exact solution is found. For a lot of systems this already happens
/// after a very low number of iterations.
/// Every iteration has complexity O(n^2) and after n iterations the
/// exact solution is found. However if this solution is needed, the other
/// methods, as for xample solveSymmSystem are more suitable.
/// 
/// This algorithm stops after the maximum number of iterations is
/// exceeded or after the max-norm of the residual \f$ r_k= Ax_k-b\f$ is 
/// smaller than epsilon. The reuslt is stored in b afterwars
/// 
/// \param A the positive semi-definite n x n-Matrix
/// \param b the right hand side which also stores the final solution
/// \param epsilon stopping criterium for the residual
/// \param maxIterations the maximum number of iterations
template<class MatT, class VecT>
void approxSolveSymmSystemInPlace(
	matrix_expression<MatT> const& A,
	vector_expression<VecT>& b,
	double epsilon = 1.e-10,
	unsigned int maxIterations = 0
){
	SIZE_CHECK(A().size1()==A().size2());
	SIZE_CHECK(A().size1()==b().size());
	vector<typename VecT::value_type> x(b.size(),0.0);
	approxSolveSymmSystem(A,x,b);
	swap(x,b);
}

/** @}*/
}}


#include "Impl/solveSystem.inl"
#endif
