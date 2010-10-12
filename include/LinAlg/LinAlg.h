//===========================================================================
/*!
 *  \file LinAlg.h
 *
 *  \brief Some operations for matrices.
 *
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
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
 *  \par Project:
 *      LinAlg
 *
 *
 *
 *  This file is part of LinAlg. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 *
 */
//===========================================================================

#ifndef LINALG_H
#define LINALG_H

#include "Array/Array2D.h"


//! Sorts the eigenvalues in vector "dvecA" in descending order and the corresponding
//! eigenvectors in matrix "vmatA".
void eigensort
(
	Array2D< double >& vmatA,
	Array  < double >& dvecA
);

//! Calculates the eigenvalues and the normalized eigenvectors of the
//! symmetric matrix "amatA" using the Jacobi method.
void eigensymmJacobi
(
	Array2D< double >& amatA,
	Array2D< double >& vmatA,
	Array  < double >& dvecA
);

//! Calculates the eigenvalues and the normalized
//! eigenvectors of the symmetric matrix "amatA" using a modified
//! Jacobi method.
void eigensymmJacobi2
(
	Array2D< double >& amatA,
	Array2D< double >& vmatA,
	Array  < double >& dvecA
);



//! Calculates the eigenvalues and the normalized
//! eigenvectors of a symmetric matrix "amatA" using the Givens
//! and Householder reduction, however, "hmatA" contains intermediate results after application.
void eigensymm_intermediate
(
	const Array2D< double >& amatA,
	Array2D <double >& hmatA,
	Array2D< double >& vmatA,
	Array  < double >& dvecA
);

//! Used as frontend for
//! #eigensymm(const Array2D<double> &amatA,Array2D<double> &vmatA,Array<double> &dvecA,Array<double> &odvecA)
//! for calculating the eigenvalues and the normalized eigenvectors of a symmetric matrix "amatA" using the Givens
//! and Householder reduction without corrupting "amatA" during application. Each time this frontend is
//! called additional memory is allocated for intermediate results.

void eigensymm
(
	const Array2D< double >& amatA,
	Array2D< double >& vmatA,
	Array  < double >& dvecA
);


//! Used as another frontend for
//! #eigensymm(const Array2D<double> &amatA,Array2D<double> &vmatA,Array<double> &dvecA) for calculating
//! the eigenvalues and the normalized eigenvectors of a symmetric matrix "amatA" using the Givens
//! and Householder reduction without corrupting "A" during application. Each time this frontend is
//! called additional memory is allocated for intermediate results.

void eigensymm
(
	const Array< double >& A,
	Array< double >& G,
	Array< double >& l
);


//! Calculates the eigenvalues and the normalized eigenvectors of a symmetric matrix "amatA" using the Givens
//! and Householder reduction. This method works without corrupting "amatA" during application by demanding
//! another Array 'odvecA' as an algorithmic buffer instead of using the last row of 'amatA' to store
//! intermediate algorithmic results.

void eigensymm
(
	const Array2D< double >& amatA,
	Array2D< double >& vmatA,
	Array  < double >& dvecA,
	Array  < double >& odvecA
);


//! Used as frontend for
//! #eigensymm(const Array2D<double> &amatA,Array2D<double> &vmatA,Array<double> &dvecA,Array<double> &odvecA)
//! for calculating the eigenvalues and the normalized eigenvectors of a symmetric matrix "amatA" using the
//! Givens and Householder reduction without corrupting "A" during application.

void eigensymm
(
	const Array< double >& A,
	Array< double >& G,
	Array< double >& l,
	Array< double >& od
);

// eigen computes the eigenvalues of an arbitrary (non-symmetric)
// matrix A and stores the unsorted eigenvalues in vr and vi (vr real
// parts, vi complex parts)

void eigen(Array<double> A, Array<double> & vr, Array<double> & vi);

//===========================================================================
/*!
 *  \brief lower triangular Cholesky decomposition
 *
 *  Given an \f$ m \times m \f$ symmetric positive definite matrix
 *  \f$M\f$, compute the lower triangular matrix \f$C\f$ such that \f$
 *  M=CC^T \f$
 *
 *      \param  M \f$ m \times m \f$ matrix, which must be symmetric and positive definite
 *      \param	C \f$ m \times m \f$ matrix, which stores the Cholesky factor
 *      \return none
 *
 *  \author  T. Suttorp and C. Igel
 *  \date    2008
 *
 *  \par Status
 *      stable
 *
 */

void CholeskyDecomposition(const Array2D< double >& M, Array2D< double >& C);



//! Calculates the relative error of eigenvalue  no. "c".
double eigenerr
(
	const Array2D< double >& amatA,
	const Array2D< double >& vmatA,
	const Array  < double >& dvecA,
	unsigned c
);

//! Determines the rank of the symmetric matrix "amatA".
unsigned sym_rank
(
	const Array2D< double >& amatA,
	const Array2D< double >& vmatA,
	const Array  < double >& dvecA
);

//! Calculates the determinant of the symmetric matrix "amatA".
double detsymm
(
	Array2D< double >& amatA,
	Array2D< double >& vmatA,
	Array  < double >& dvecA
);

//! Calculates the log of the determinant of the symmetric matrix "amatA".
double logdetsymm
(
	Array2D< double >& amatA,
	Array2D< double >& vmatA,
	Array  < double >& dvecA
);

//! Calculates the rank of the symmetric matrix "amatA", its eigenvalues and
//! eigenvectors.
unsigned rankDecomp
(
        Array2D< double >& amatA,
	Array2D< double >& vmatA,
	Array2D< double >& hmatA,
	Array  < double >& dvecA
);

//! Given "m" classes of data, the covariances between
//! the values within one class and the covariances
//! between all the classes, this method calculates
//! the transformation matrix that will project the
//! data in a way, that maximum separation of the
//! different classes is given.
void discrimAnalysis
(
	Array2D< double >& betweenCovarA,
	Array2D< double >& withinCovarA,
	Array2D< double >& transMatA,
	Array  < double >& dvecA,
	unsigned& m
);


//! Given the correlations of the n-dimensional data vector "x"
//! and the m-dimensional data vector "y" and also given
//! their mean values, this function summarizes the data by
//! finding a linear mapping that will approximate the data.
void linearRegress
(
	Array2D< double >& cxxMatA,
	Array2D< double >& cxyMatA,
	Array  < double >& mxVecA,
	Array  < double >& myVecA,
	Array2D< double >& amatA,
	Array  < double >& bvecA,
	Array  < double >& dvecA
);


//! Determines the numerical rank of a rectangular matrix "amatA",
//! when a singular value decomposition for "amatA" has taken place
//! before.
unsigned svdrank
(
	const Array2D< double >& amatA,
	Array2D< double >& umatA,
	Array2D< double >& vmatA,
	Array  < double >& wvecA
);


//! \par
//! Determines the singular value decomposition of a rectangular
//! matrix "amatA".
//!
//! \par
//! See also: <a ref="http://en.wikipedia.org/wiki/Singular_value_decomposition">Wikipedia</a>
void svd
(
	const Array2D< double >& amatA,
	Array2D< double >& umatA,
	Array2D< double >& vmatA,
	Array  < double >& wvecA,
	unsigned maxIterations = 200,
	bool ignoreThreshold = true
);



//! Sorts the singular values in vector "wvecA" by descending order.
void svdsort
(
	Array2D< double >& umatA,
	Array2D< double >& vmatA,
	Array  < double >& wvecA
);



//===========================================================================
/*!
 *  \brief Transpose of matrix "v".
 *
 *  Given the matrix \em v, the transposed matrix is created, i.e.
 *  the rows become columns and vice versa:
 *  \f[
 *      \left(
 *      \begin{array}{*{4}{c}}
 *          1  & 2  & 3  & 4\\
 *          5  & 6  & 7  & 8\\
 *          9  & 10 & 11 & 12\\
 *          13 & 14 & 15 & 16\\
 *      \end{array}
 *      \right)
 *      \longrightarrow
 *      \left(
 *      \begin{array}{*{4}{c}}
 *          1 & 5 & 9  & 13\\
 *          2 & 6 & 10 & 14\\
 *          3 & 7 & 11 & 15\\
 *          4 & 8 & 12 & 16\\
 *      \end{array}
 *      \right)
 *  \f]
 *  The original matrix \em v will not be modified.
 *
 *      \param  v matrix, that will be transposed
 *      \return the transposed matrix
 *      \throw check_exception the type of the exception will be
 *             "size mismatch" and indicates that \em v is not
 *             2-dimensional
 *
 *  \example linalg_simple_test.cpp
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      2002-01-23, ra: <br>
 *      Formerly worked only for square matrices - fixed.
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
Array< T > transpose(const Array< T >& v)
{
	SIZE_CHECK(v.ndim() == 2)
	Array< T > z;
	z.resize(v.dim(1), v.dim(0));
	for (unsigned i = v.dim(0); i--;)
		for (unsigned j = v.dim(1); j--;)
			z(j, i) = v(i, j);
	return z;
}


//===========================================================================
/*!
 *  \brief Creates a new matrix with vector "v" lying on the
 *         diagonal.
 *
 *  Given the vector \em v, a new matrix is created. The diagonal
 *  of the new matrix adapt the values of \em v and all other
 *  other matrix positions are set to zero.<br>
 *  Example:
 *
 *  \f[
 *      \left(
 *      \begin{array}{*{4}{c}}
 *          1 \\
 *          2 \\
 *          3 \\
 *          4 \\
 *      \end{array}
 *      \right)
 *      \longrightarrow
 *      \left(
 *      \begin{array}{*{4}{c}}
 *          1 & 0 & 0 & 0\\
 *          0 & 2 & 0 & 0\\
 *          0 & 0 & 3 & 0\\
 *          0 & 0 & 0 & 4\\
 *      \end{array}
 *      \right)
 *  \f]
 *
 *      \param  v vector with values for the matrix diagonal
 *      \return the new matrix with \em v as diagonal
 *      \throw check_exception the type of the exception will be
 *             "size mismatch" and indicates that \em v is not
 *             one-dimensional
 *
 *  \example linalg_simple_test.cpp
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
Array< T > diagonal(const Array< T >& v)
{
	SIZE_CHECK(v.ndim() == 1)
	Array< T > z(v.nelem(), v.nelem());
	z = 0;
	for (unsigned i = v.nelem(); i--;)
		z(i, i) = v.elem(i);
	return z;
}


//===========================================================================
/*!
 *  \brief Evaluates the sum of the values at the diagonal of
 *         matrix "v".
 *
 *  Example:
 *  \f[
 *      \left(
 *      \begin{array}{*{4}{c}}
 *          {\bf 1} & 5       & 9        & 13\\
 *          2       & {\bf 6} & 10       & 14\\
 *          3       & 7       & {\bf 11} & 15\\
 *          4       & 8       & 12       & {\bf 16}\\
 *      \end{array}
 *      \right)
 *      \longrightarrow 1 + 6 + 11 + 16 = 34
 *  \f]
 *
 *      \param  v square matrix
 *      \return the sum of the values at the diagonal of \em v
 *      \throw check_exception the type of the exception will be
 *             "size mismatch" and indicates that \em v is
 *             not a square matrix
 *
 *  \example  linalg_simple_test.cpp
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
T trace(const Array< T >& v)
{
	SIZE_CHECK(v.ndim() == 2 && v.dim(0) == v.dim(1))

	T t(v(0, 0));
	for (unsigned i = 1; i < v.dim(0); ++i)
		t += v(i, i);
	return t;
}


//! Calculates the mean vector of array "x".
Array< double > mean(const Array< double >& x);


//! Calculates the variance vector of array "x".
Array< double > variance(const Array< double >& x);


//! Calculates the angle between the vectors "x" and "y".
double angle(const Array< double >& x, const Array< double >& y);


//! Calculates the coefficient of correlation of the data
//! vectors "x" and "y".
double corrcoef(const Array< double >& x, const Array< double >& y);


//! Calculates the coefficient of correlation matrix of the data
//! vectors stored in matrix "x".
Array< double > corrcoef(const Array< double >& x);


//! Calculates the mean and variance values of matrix "x".
void meanvar
(
	const Array< double >& x,
	Array< double >&,
	Array< double >&
);

//! Calculates the mean and variance values of 1d-arrays p(x)
void meanvar
(
	const Array< double >& pxA,
	const Array< double >& xA,
	double &mA,
	double &vA,
	const int startA = -1,
	const int endA = -1
);


//! Calculates the covariance between the data vectors "x" and "y".
double covariance(const Array< double >& x, const Array< double >& y);


//! Calculates the covariance matrix of the data vectors stored in
//! matrix "x".
Array< double > covariance(const Array< double >& x);


//! Returns the generalized inverse matrix of input matrix
//! "A" by using singular value decomposition. Used as frontend
//! for metod #g_inverse when using type "Array" instead of
//! "Array2D".
Array< double > invert(const Array< double >&);


//! Multiplies two 2D matrices
void matMat(Array2D<double> &A, const Array2D<double> &B, const Array2D<double> &C);

//! Returns A = BC, where C is viewed as a column vector
void matColVec(Array<double> &A, const Array2D<double> &B, const Array<double> &C);

//! Returns A = BC, where C is viewed as a column vector
void matColVec(ArrayReference<double> A, const Array2D<double> &B, const ArrayReference<double> C);

//! Returns \f$ A = B C_i \f$ , where \f$ C_i \f$ is a column of the matrix C.
void matColVec(Array<double> &A, const Array2D<double> &B, const Array<double> &C, unsigned int index);

//! Returns the scalar ABC
double vecMatVec(const Array<double> &A, const Array2D<double> &B, const Array<double> &C);

//! Returns the scalar \f$ A_i B C_j \f$
double vecMatVec(const Array<double> &A, unsigned int i, const Array2D<double> &B, const Array<double> &C, unsigned int j);


//! Inverts a symmetric matrix
void invertSymm(Array2D<double> &I, const Array2D< double >& A);

//! Inverts a symmetric positive definite matrix
void invertSymmPositiveDefinite(Array2D<double> &I, const Array2D< double >& ArrSymm);

//! Calculates the generalized inverse matrix of input matrix "amatA".
unsigned g_inverse
(
	const Array2D< double >& amatA,
	Array2D< double >& bmatA,
	unsigned maxIterations = 200,
	double tolerance = 1e-10,
	bool ignoreThreshold = true
);

//! Returns the generalized inverse matrix of input matrix using Cholesky decomposition
void g_inverseCholesky(const Array2D< double >& A, Array2D< double >& outA, double thresholdFactor =  1e-9); 

//! Returns the generalized inverse matrix of input matrix \f$ A \f$ using Cholesky decomposition assuming that \f$ A^T A \f$ has full rank
void g_inverseMoorePenrose(const Array2D< double >& A, Array2D< double >& outA);

#endif  // LINALG_H

