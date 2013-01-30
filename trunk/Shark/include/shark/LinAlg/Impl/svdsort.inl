//===========================================================================
/*!
 *  \file svdsort.inl
 *
 *  \brief Used for sorting singular values and the orthogonal
 *         matrices \em U and \em V after a singular value
 *         decomposition of an input matrix \em A.
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
#ifndef SHARK_LINALG_SVDSORT_INL
#define SHARK_LINALG_SVDSORT_INL

#include <cmath>

//===========================================================================
/*!
 *  \brief Sorts the singular values in vector "wvecA" by descending order.
 *
 *  For a singular value decomposition defined as
 *
 *  \f$
 *      A = UWV^T
 *  \f$
 *
 *  the resulting orthogonal matrices \em U and \em V and the singular
 *  values in \em W can be sorted in a way, that the singular values
 *  are given in descending order, when leaving the function.
 *
 *      \param  umatA The \f$ m \times n \f$ matrix \em U.
 *      \param  vmatA The \f$ n \times n \f$ matrix \em V.
 *      \param  wvecA n-dimensional vector containing the singular
 *                    values.
 *      \return       none.
 *      \throw SharkException the type of the exception will
 *             be "size mismatch" and indicates that \em wvecA is not
 *             one-dimensional
 *
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
 *  \sa svd.cpp
 *
 */
template<class MatrixU,class VectorT>
void shark::svdsort
(
	MatrixU& umatA,
	MatrixU& vmatA,
	VectorT& wvecA
)
{

	unsigned m = umatA.size1(); /* rows */
	unsigned n = umatA.size2(); /* cols */

	unsigned i, j, k;
	double   p;

	for (i = 0; i < n - 1; i++)
	{
		p = wvecA (k = i);
		//find largest remaining singular value
		for (j = i + 1; j < n; j++)
		{
			if (wvecA (j) >= p)
			{
				p = wvecA (k = j);
			}
		}

		if (k != i) {
		  //switch current and largest value
			wvecA (k) = wvecA (i);
			wvecA (i) = p;
			//switch corresponding vectors
			for (j = 0; j < n; j++) {
				p         = vmatA (j, i);
				vmatA (j, i) = vmatA (j, k);
				vmatA (j, k) = p;
			}

			for (j = 0; j < m; j++) {
				p         = umatA (j, i);
				umatA (j, i) = umatA (j, k);
				umatA (j, k) = p;
			}
		}
	}
}
#endif
