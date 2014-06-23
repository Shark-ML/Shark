//===========================================================================
/*!
 *  \brief Used to calculate the eigenvalues and the eigenvectors of a
 *         symmetric matrix.
 *
 *  Here the eigenvectors and eigenvalues are calculated by using
 *  Givens and Householder reduction of the matrix to tridiagonal form.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *
 *  <BR><HR>
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
 */
//===========================================================================
#ifndef SHARK_LINALG_EIGENSYMM_INL
#define SHARK_LINALG_EIGENSYMM_INL

//===========================================================================
/*!
 *  \brief Used as frontend for
 *  eigensymm for calculating the eigenvalues and the normalized eigenvectors of a symmetric matrix
 *  'A' using the Givens and Householder reduction. Each time this frontend is called additional
 *  memory is allocated for intermediate results.
 *
 *
 * \param A \f$ n \times n \f$ matrix, which must be symmetric, so only the bottom triangular matrix must contain values.
 * \param G \f$ n \times n \f$ matrix with the calculated normalizedeigenvectors, each column will contain one eigenvector.
 * \param l n-dimensional vector with the calculated eigenvalues in descending order.
 * \return none.
 *
 * \throw SharkException
 */
template<class MatrixT,class MatrixU,class VectorT>
void shark::blas::eigensymm
(
	const MatrixT& A,
	MatrixU& G,
	VectorT& l
)
{
	unsigned int n = A.size1();
	SIZE_CHECK(A.size2() == n);

	G.resize(n,n);
	l.resize(n);
	
	//
	// special case n = 1
	//
	if (n == 1) {
		vmatA( 0 ,  0 ) = 1;
		dvecA( 0 ) = amatA( 0 ,  0 );
		return;
	}

	//
	// copy matrix
	//
	for (unsigned i = 0; i < n; i++) {
		for (j = 0; j <= i; j++) {
			vmatA(i, j) = amatA(i, j);
		}
	}

	VectorT od(n);

	eigensymm(A, G, l, od);
}


//===========================================================================
/*!
 *  \brief Calculates the eigenvalues and the normalized
 *  eigenvectors of a symmetric matrix "amatA" using the Givens
 *  and Householder reduction without corrupting  "amatA" during application.
 *
 *  Given a symmetric \f$ n \times n \f$ matrix \em A, this function
 *  calculates the eigenvalues \f$ \lambda \f$ and the eigenvectors \em x,
 *  defined as
 *
 *  \f$
 *      Ax = \lambda x
 *  \f$
 *
 *  where \em x is a one-column matrix and the matrix multiplication
 *  is used for \em A and \em x.
 *  Here, the Givens reduction as a modification of the Jacobi method
 *  is used. Instead of trying to reduce the
 *  matrix all the way to diagonal form, we are content to stop
 *  when the matrix is tridiagonal. This allows the function
 *  to be carried out in a finite number of steps, unlike the
 *  Jacobi method, which requires iteration to convergence.
 *  So in comparison to the Jacobi method, this function is
 *  faster for matrices with an order greater than 10.
 *
 *  \param amatA \f$ n \times n \f$ matrix, which must be symmetric, so only the bottom triangular matrix must contain values.
 *  \param vmatA \f$ n \times n \f$ matrix with the calculated normalized eigenvectors, each column will contain an eigenvector.
 *  \param  dvecA n-dimensional vector with the calculated eigenvalues in descending order.
 *  \param  odvecA n-dimensional vector with the calculated offdiagonal of the Householder transformation.
 *
 *  \return       none.
 *
 *  \throw SharkException
 *
 */
template<class MatrixT,class MatrixU,class VectorT>
void shark::blas::eigensymm
(
	const MatrixT& amatA,
	MatrixU& vmatA,
	VectorT& dvecA,
	VectorT& odvecA
)
{

	

	

}
#endif
