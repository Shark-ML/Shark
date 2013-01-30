//===========================================================================
/*!
 *  \file rankDecomp.inl
 *
 *  \brief Used to calculate the rank, the eigenvalues and eigenvectors
 *         of the symmetric matrix "amatA".
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Copyright (c) 1998-2001:
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

#ifndef SHARK_LINALG_RANKDECOMP_INL
#define SHARK_LINALG_RANKDECOMP_INL
#include <cmath>

//===========================================================================
/*!
 *  \brief Calculates the rank of the symmetric matrix "amatA",
 *         its eigenvalues and eigenvectors.
 *
 *  Determines the rank of matrix \em amatA and additionally
 *  calculates the eigenvalues and eigenvectors of the matrix.
 *  Empty eigenvalues (i.e. eigenvalues equal to zero) are
 *  set to the greatest calculated eigenvalue and each
 *  eigenvector \f$ x_j \f$ is scaled by multiplying
 *  it with the scalar value \f$ \frac{1}{\sqrt{\lambda_j}} \f$.
 *
 *      \param  amatA \f$ n \times n \f$ matrix, which is symmetric, so
 *                    only the bottom triangular matrix must contain
 *                    values. At the end of the function \em amatA
 *                    always contains the full matrix.
 *      \param	vmatA \f$ n \times n \f$ matrix, which will
 *                    contain the scaled eigenvectors.
 *      \param  hmatA \f$ n \times n \f$ matrix, that is used to store
 *                    intermediate  results.
 *	\param  dvecA n-dimensional vector with the calculated
 *                    eigenvalues in descending order.
 *      \return       The rank of matrix \em amatA.
 *      \throw SharkException the type of the exception will be
 *             "size mismatch" and indicates that \em amatA
 *             is not a square matrix
 *
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      2003/10/02 by S. Wiegand
 *      due to name change of 'eigensymm';
 *
 *  \par Status
 *      stable
 *
 */
template<class MatrixT,class MatrixU,class MatrixV,class VectorT>
unsigned shark::rankDecomp
(
	MatrixT& amatA,
	MatrixU& vmatA,
	MatrixV& hmatA,
	VectorT& dvecA
)
{
	SIZE_CHECK(amatA.size1() == amatA.size2());
	unsigned n = amatA.size1();

	vmatA.resize(n,n, false);
	dvecA.resize(n, false);

	/*
	 * calculate eigenvalues (save secondary diagonals)
	 */
	for (size_t i = 0; i + 1 < n; i++)
		for (size_t j = i + 1; j < n; j++)
			amatA(i, j) = amatA(j, i);

	//eigensymm( amatA, vmatA, dvecA );
	eigensymm_intermediate(amatA, hmatA, vmatA, dvecA);

	for (size_t i = 0; i + 1 < n; i++)
		for (size_t j = i + 1; j < n; j++)
			amatA(j, i) = amatA(i, j);

	/*
	 * determine numercial rank
	 */
	unsigned r = rank(hmatA, vmatA, dvecA);

	/*
	 * set eigenvalues of arbitrary constant 0
	 */
	for (size_t i = r; i < n; i++) dvecA(i) = dvecA(0);

	/*
	 * scale eigenvalues
	 */
	if (r)
		for (size_t j = n; j--;)
		{
			double dt = 1. / std::sqrt(dvecA( j ));

			for (size_t i = n; i--;) vmatA( i ,  j ) *= dt;
		}

	return r;
}
#endif
