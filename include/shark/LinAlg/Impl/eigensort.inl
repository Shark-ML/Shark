/**
*
*  \brief Inline implementation of eigenvalue/-vector sorting
*
* \ingroup shark_globals
* 
*  \par Copyright (c) 1998-2011:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
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


#ifndef SHARK_LINALG_EIGENSORT_INL
#define SHARK_LINALG_EIGENSORT_INL


//===========================================================================
/*!
 *  \brief Sorts the eigenvalues in vector "dvecA" and the corresponding
 *         eigenvectors in matrix "vmatA".
 * 
 *  Given the matrix \em vmatA of eigenvectors and the vector
 *  \em dvecA of corresponding eigenvalues, the values in \em dvecA will
 *  be sorted by descending order and the eigenvectors in
 *  \em vmatA will change their places in a way, that at the
 *  end of the function an eigenvalue at position \em j of
 *  vector \em dvecA will belong to the eigenvector
 *  at column \em j of matrix \em vmatA.
 *  If we've got for example the following result after calling the function:
 *
 *  \f$
 *      \begin{array}{*{3}{r}}
 *          v_{11} & v_{21} & v_{31}\\
 *          v_{12} & v_{22} & v_{32}\\
 *          v_{13} & v_{23} & v_{33}\\
 *          & & \\
 *          v_1 & v_2 & v_3\\
 *      \end{array}
 *  \f$
 *
 *  then eigenvalue \f$ v_1 \f$ has the corresponding eigenvector
 *  \f$ ( v_{11}\ v_{12}\ v_{13} ) \f$ and \f$ v_1 > v_2 > v_3 \f$.
 *
 *
 *      \param	vmatA \f$ n \times n \f$ matrix with eigenvectors (each column
 *                    contains an eigenvector, corresponding to
 *                    one eigenvalue).
 *	\param  dvecA n-dimensional vector with eigenvalues, will
 *                    contain the eigenvalues in descending order
 *                    when returning from the function.
 *      \return       none.
 *      \throw SharkException the type of the exception will be
 *             "size mismatch" and indicates that \em dvecA
 *             is not one-dimensional or that the number of
 *             rows or the number of columns in \em vmatA
 *             is different from the number of values
 *             in \em dvecA
 *
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 */
template<class MatrixT,class VectorT>
void shark::blas::eigensort
(
	MatrixT& vmatA,
	VectorT& dvecA
)
{
	SIZE_CHECK
	(
		dvecA.size() == vmatA.size1() &&
		dvecA.size() == vmatA.size2()
	);

	unsigned n = dvecA.size();
	unsigned i, j, l;//l: position of largest remainig eigenvalue
	double t;//largest remaining eigenvalue

	//
	// sort eigen values
	//
	for (i = 0; i < n - 1; i++)
	{
		t = dvecA( l = i );
		//find largest remaining eigenvalue
		for (j = i + 1; j < n; j++) {
			if (dvecA( j ) >= t) {
				t = dvecA( l = j );
			}
		}

		if (l != i) {
		        //switch position of i's eigenvalue and the largest remaining eigenvalue
			dvecA( l ) = dvecA( i );
			dvecA( i ) = t;
			//switch postions of corresponding eigenvectors
			for (j = 0; j < n; j++) {
				t           = vmatA( j , i );
				vmatA( j , i ) = vmatA( j , l );
				vmatA( j , l ) = t;
			}
		}
	}
}

#endif
