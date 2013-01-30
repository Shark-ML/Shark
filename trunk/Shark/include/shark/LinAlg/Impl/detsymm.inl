//===========================================================================
/*!
 *  \brief Used to calculate the determinant, the eigenvalues and eigenvectors
 *         of the symmetric matrix "amatA".
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Copyright (c) 1998-2000:
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
#ifndef SHARK_LINALG_DETSYMM_INL
#define SHARK_LINALG_DETSYMM_INL
#include <cmath>

//===========================================================================
/*!
 *  \brief Calculates the determinate of the symmetric matrix "amatA".
 *
 *  Calculates the determinate of matrix \em amatA by using its
 *  \em n eigenvalues \f$ x_j \f$ that first will be calculated.
 *  The determinate is then given as:
 *
 *  \f$
 *  \prod_{j=1}^n x_j
 *  \f$
 *
 *      \param  amatA \f$ n \times n \f$ matrix, which is symmetric, so
 *                    only the bottom triangular matrix must contain
 *                    values. At the end of the function \em amatA
 *                    always contains the full matrix.
 *      \param	vmatA \f$ n \times n \f$ matrix, that will
 *                    contain the scaled eigenvectors at the
 *                    end of the function.
 *	\param  dvecA n-dimensional vector that will contain
 *                    the eigenvalues at the end of the
 *                    function.
 *      \return       The determinate of matrix \em amatA.
 *      \throw SharkException the type of the eception will be
 *             "size mismatch" and indicates that \em amatA is
 *             not a square matrix
 *
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
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
template<class MatrixT,class MatrixU,class VectorT>
double shark::detsymm
(
	MatrixT& amatA,
	MatrixU& vmatA,
	VectorT& dvecA
)
{
	SIZE_CHECK(amatA.size1() == amatA.size2());

	unsigned n = amatA.size1();

	// Fill upper triangular matrix:
	for (size_t i = 0; i + 1 < n; i++) {
		for (size_t j = i + 1; j < n; j++) {
			amatA(i, j) = amatA(j, i);
		}
	}

	// Calculate eigenvalues and use intermediate results stored in hmatA
	//eigensymm( amatA, vmatA, dvecA );
	MatrixU hmatA(amatA);
	eigensymm_intermediate(amatA, hmatA, vmatA, dvecA);

	for (size_t i = 0; i + 1 < n; i++)
	{
		for (size_t j = i + 1; j < n; j++)
		{
			hmatA(j, i) = hmatA(i, j);
			amatA(j, i) = hmatA(j, i);
			amatA(i, j) = hmatA(i, j);
		}
	}

	// Calculate determinate as product of eigenvalues:
	double det = 1;
	for (size_t i = 0; i < n; i++)
	{
		det *= dvecA(i);
	}
	return det;
}


//===========================================================================
/*!
 *  \brief Calculates  logarithm of the determinant of the symmetric matrix "amatA".
 *
 *  Calculates the logarithm of the determinate of matrix \em amatA by
 *  using its \em n eigenvalues \f$ x_j \f$ that first will be
 *  calculated.  The determinate is then given as:
 *
 *  \f$
 *  \prod_{j=1}^n x_j
 *  \f$
 *
 *      \param  amatA \f$ n \times n \f$ matrix, which is symmetric, so
 *                    only the bottom triangular matrix must contain
 *                    values. At the end of the function \em amatA
 *                    always contains the full matrix.
 *      \param	vmatA \f$ n \times n \f$ matrix, that will
 *                    contain the scaled eigenvectors at the
 *                    end of the function.
 *	\param  dvecA n-dimensional vector that will contain
 *                    the eigenvalues at the end of the
 *                    function.
 *      \return       The logarithm of the determinate of matrix \em amatA.
 *      \throw SharkException the type of the eception will be
 *             "size mismatch" and indicates that \em amatA is
 *             not a square matrix
 *
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *
 *  \author  C.Igel
 *  \date    2007
 *
 *
 *  \par Status
 *      stable
 *
 */
template<class MatrixT,class MatrixU,class VectorT>
double shark::logdetsymm
(
	MatrixT& amatA,
	MatrixU& vmatA,
	VectorT& dvecA
)
{
	SIZE_CHECK(amatA.size1() == amatA.size2());

	unsigned n = amatA.size1();

	// Fill upper triangular matrix:
	for (size_t i = 0; i + 1 < n; i++) {
		for (size_t j = i + 1; j < n; j++) {
			amatA(i, j) = amatA(j, i);
		}
	}

	// Calculate eigenvalues and use intermediate results stored in hmatA
	//eigensymm( amatA, vmatA, dvecA );
	MatrixU hmatA(amatA);
	eigensymm_intermediate(amatA, hmatA, vmatA, dvecA);

	for (size_t i = 0; i + 1 < n; i++)
	{
		for (size_t j = i + 1; j < n; j++)
		{
			hmatA(j, i) = hmatA(i, j);
			amatA(j, i) = hmatA(j, i);
			amatA(i, j) = hmatA(i, j);
		}
	}

	// Calculate logarithmic determinate
	double det = 0;
	for (size_t i = 0; i < n; i++)
	{
		det += std::log(dvecA(i));
	}
	return det;
}
#endif
