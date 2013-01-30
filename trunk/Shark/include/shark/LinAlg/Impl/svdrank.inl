//===========================================================================
/*!
 *  \file svdrank.inl
 *
 *  \brief Determines the numerical rank of a rectangular matrix,
 *         when a singular value decomposition for this matrix has taken
 *         place before.
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

#ifndef SHARK_LINALG_SVDRANK_INL
#define SHARK_LINALG_SVDRANK_INL
#include <cmath>

#ifdef _WINDOWS
// disable warning C2055: unreferenced formal parameter
#pragma warning(disable: 4100)
#endif

//===========================================================================
/*!
 *  \brief Determines the numerical rank of a rectangular matrix "amatA",
 *         when a singular value decomposition for "amatA" has taken place
 *         before.
 *
 *  For a singular value decomposition defined as
 *
 *  \f$
 *      A = UWV^T
 *  \f$
 *
 *  the resulting orthogonal matrices \em U and \em V and the singular
 *  values in \em W sorted by descending order are used to determine
 *  the rank of input matrix \em A.
 *
 *      \param  amatA The \f$ m \times n \f$ input matrix \em A, with
 *                    \f$ m \geq n \f$.
 *      \param  umatA The \f$ m \times n \f$ column-orthogonal matrix \em U.
 *      \param  vmatA The \f$ n \times n \f$ orthogonal matrix \em V.
 *      \param  wvecA n-dimensional vector containing the singular
 *                    values in descending order.
 *      \return       none.
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
template<class MatrixT,class MatrixU,class VectorT>
unsigned shark::svdrank
(
	const MatrixT& amatA,
	MatrixU& umatA,
	MatrixU& vmatA,
	VectorT& wvecA
)
{
	unsigned n = amatA.size2();

	/*
	  determine numerical rank: if the last singular values are < 0, the
       	  numerical error for the values > 0 is at least as large as the
	  abslolute value of the last eigenvalue, thus values below this
	  threshold are to be discarded. Additional thresholds are the relative
	  machine accuracy and the relative error in the singular values
	 */
	unsigned r;
	for (r = 0; (r < n) && (wvecA( r ) > 0.); r++) ;

	double t = r < n ? fabs(wvecA( n-1 )) : 0.0;
	r = 0;
	double s = 0.0;
	while (r < n     &&
			wvecA( r ) > t &&
			wvecA( r ) + s > s)
		s += wvecA( r++ );

	return r;
}
#endif
