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

#ifndef SHARK_LINALG_SOLVE_SYSTEM_H
#define SHARK_LINALG_SOLVE_SYSTEM_H

#include <shark/LinAlg/solveTriangular.h>

namespace shark{ namespace blas{
/// \brief System of linear equations solver.
///
///Solves asystem of linear equations
///Ax=b 
///for x, using LU decomposition and
///backward substitution. This Method is in
///no way optimized for sparse matrices.
///Be aware, that the matrix must have full rank!
template<class MatT,class Vec1T,class Vec2T>
void solveSystem(
	matrix_expression<MatT> const& A, 
	vector_expression<Vec1T>& x,
	vector_expression<Vec2T> const& b
);

/// \brief System of linear equations solver.
///
///Solves multiple systems of linear equations
///Ax_1=b_1
///Ax_2=b_2
///...
///=>AX=B
///for X, using LU decomposition and
///backward substitution.
///Note, that B=(b_1,...,b_n), so the right hand sides are stored as columns
///This Method is in no way optimized for sparse matrices.
///Be aware, that the matrix must have full rank!
template<class MatT,class Vec1T,class Vec2T>
void solveSystem(
	matrix_expression<MatT> const& A, 
	matrix_expression<Vec1T>& x,
	matrix_expression<Vec2T> const& b
);

/// \brief System of symmetric linear equations solver. The result is stored in b
///
///Solves a system of linear equations
///Ax=b 
///for x, using Cholesky decomposition and
///backward substitution. and stores the result in b.
/// A must be symmetric.
///This method is in no way optimized for sparse matrices.
///Be aware, that the matrix must have full rank!
template<class System, class MatT,class VecT>
void solveSymmSystemInPlace(
	matrix_expression<MatT> const& A,
	vector_expression<VecT>& b
);

/// \brief System of symmetric linear equations solver.
///
///Solves multiple systems of linear equations
///Ax_1=b_1
///Ax_1=b_2
///...
///=>AX=B
///or XA=B
///for X, using cholesky decomposition and
///backward substitution. The first template parameter is used 
///to decide which type of system is solved
///Note, that B=(b_1,...,b_n), so the right hand sides are stored as columns.
///A must be symmetric.
///This Method is in no way optimized for sparse matrices.
///Be aware, that the matrix must have full rank!
///Also the result is stored in B directly so it"s contents are destroyed.
///@param A the system matrix A
///@param B the right hand side of the LGS, also stores the result
template<class System, class MatT,class Mat1T>
void solveSymmSystemInPlace(
	matrix_expression<MatT> const& A,
	matrix_expression<Mat1T>& B
);

/// \brief System of symmetric linear equations solver.
///
///Solves a system of linear equations
///Ax=b 
///for x, using Cholesky decomposition and
///backward substitution.  A must be symmetric.
///This Method is in no way optimized for sparse matrices.
///Be aware, that the matrix must have full rank!
template<class System,class MatT,class Vec1T,class Vec2T>
void solveSymmSystem(
	matrix_expression<MatT> const& A, 
	vector_expression<Vec1T>& x,
	vector_expression<Vec2T> const& b
);
/// \brief System of symmetric linear equations solver.
///
///Solves multiple systems of linear equations
///Ax_1=b_1
///Ax_1=b_2
///...
///=>AX=B
///or XA = B
///for X, using cholesky decomposition and
///backward substitution. The first template parameter is used 
///to decide which type of system is solved
///Note, that B=(b_1,...,b_n), so the right hand sides are stored as columns.
///A must be symmetric.
///This Method is in no way optimized for sparse matrices.
///Be aware, that the matrix must have full rank!
///@param A the system matrix A
///@param X the stored result of the solution of LGS
///@param B the right hand side of the LGS
template<class System,class MatT,class Mat1T,class Mat2T>
void solveSymmSystem(
	matrix_expression<MatT> const& A, 
	matrix_expression<Mat1T>& X,
	matrix_expression<Mat2T> const& B
);

///\brief Approximates the solution of a linear system of equation Ax=b.
///
///Most often there is no need for the exact solution of a system of linear
///equations. Instead only a good approximation needs to be found.
///In this case an iterative method can be used which stops when
///a suitable exact solution is found. For a lot of systems this already happens
///after a very low number of iterations.
///Every iteration has complexity O(n^2) and after n iterations the
///exact solution is found. However if this solution is needed, the other
///methods, as for xample solveSymmSystem are more suitable.
///
///This algorithm stops after the maximum number of iterations is
/// exceeded or after the max-norm of the residual \f$ r_k= Ax_k-b\f$ is  
/// smaller than epsilon
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
	ensureSize(x,dim);
	zero(x);
	
	unsigned int maxIt = (maxIterations == 0)? dim: maxIterations;
	
	typedef typename VecT::value_type value_type;
	vector<value_type> r=b;
	vector<value_type> rnext(dim);
	vector<value_type> p=b;
	vector<value_type> Ap(dim);
	
	for(std::size_t i = 0; i != maxIt; ++i){
		fast_prod(A,p,Ap);
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

///\brief Approximates the solution of a linear system of equation Ax=b, storing the solution in b.
///
///Most often there is no need for the exact solution of a system of linear
///equations. Instead only a good approximation needs to be found.
///In this case an iterative method can be used which stops when
///a suitable exact solution is found. For a lot of systems this already happens
///after a very low number of iterations.
///Every iteration has complexity O(n^2) and after n iterations the
///exact solution is found. However if this solution is needed, the other
///methods, as for xample solveSymmSystem are more suitable.
///
///This algorithm stops after the maximum number of iterations is
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

}}
#include "Impl/solveSystem.inl"
#endif
