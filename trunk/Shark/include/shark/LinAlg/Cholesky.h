/*!
 * 
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


/// \brief Updates a covariance factor by a rank one update
///
/// Let \f$ A=LL^T \f$ be a matrix with its lower cholesky factor. Assume we want to update 
/// A using a simple rank-one update \f$ A = \alpha A+ \beta vv^T \f$. This invalidates L and
/// it needs to be recomputed which is O(n^3). instead we can update the factorisation
/// directly by performing a similar, albeit more complex algorithm on L, which can be done
/// in O(L^2). 
/// 
/// Alpha is not required to be positive, but if it is not negative, one has to be carefull
/// that the update would keep A positive definite. Otherwise the decomposition does not
/// exist anymore and an exception is thrown.
///
/// \param L the lower cholesky factor to be updated
/// \param v the update vector
/// \param alpha the scaling factor, must be positive.
/// \param beta the update factor. it Can be positive or negative
template<class Matrix,class Vector>
void choleskyUpdate(
	matrix_expression<Matrix>& L, 
	vector_expression<Vector> const& v, 
	double alpha, double beta
){
	//implementation blatantly stolen from Eigen
	std::size_t n = v().size();
	blas::vector<double> temp = v();
	double betaPrime = 1;
	double a = std::sqrt(alpha);
	for(std::size_t j=0; j != n; ++j)
	{
		double Ljj = a*L()(j,j);
		double dj = Ljj*Ljj;
		double wj = temp(j);
		double swj2 = beta*wj*wj;
		double gamma = dj*betaPrime + swj2;

		double x = dj + swj2/betaPrime;
		if (x <= 0.0)
			throw SHARKEXCEPTION("[choleskyUpdate] update makes matrix indefinite, no update available");
		double nLjj = std::sqrt(x);
		L()(j,j) = nLjj;
		betaPrime += swj2/dj;
		
		// Update the terms of L
		if(j+1 <n)
		{
			subrange(column(L,j),j+1,n) *= a;
			noalias(subrange(temp,j+1,n)) -= (wj/Ljj) * subrange(column(L,j),j+1,n);
			if(gamma == 0)
				continue;
			subrange(column(L,j),j+1,n) *= nLjj/Ljj;
			noalias(subrange(column(L,j),j+1,n))+= (nLjj * beta*wj/gamma)*subrange(temp,j+1,n);
		}
	}
}

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
