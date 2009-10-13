//===========================================================================
/*!
 *  \file eigenerr.cpp
 *
 *  \brief Used to calculate the relative error of one eigenvalue.
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
 */
//===========================================================================


#include <cmath>
#include <SharkDefs.h>
#include <LinAlg/LinAlg.h>


//===========================================================================
/*!
 *  \brief Calculates the relative error of one eigenvalue with no. "c".
 *
 *  Given a symmetric \f$ n \times n \f$ matrix \em amatA, the matrix \em vmatA
 *  of its eigenvectors and the vector \em dvecA of the corresponding
 *  eigenvalues, this function calculates the
 *  relative error of one eigenvalue, denoted by the no. \em c.
 *  If we have a \f$ n \times n \f$ matrix \em A, a matrix \em x of
 *  eigenvectors and a vector \f$ \lambda \f$ of corresponding
 *  eigenvalues, the relative error of eigenvalue
 *  no. \em c is calculated as
 *
 *  \f$
 *      \sqrt{\sum_{i=0}^n \left(\sum_{j=0}^n A(i,j) \ast x(j,c) - x(i,c)
 *      \ast \lambda(c) \right)^2}
 *  \f$
 *
 *      \param  amatA \f$ n \times n \f$ matrix, which has to be symmetric,
 *                    so only the lower
 *                    triangular matrix must contain values.
 *                    The matrix is not changed by the function.
 *      \param	vmatA \f$ n \times n \f$ matrix with normalized eigenvectors,
 *                    each column contains an eigenvector.
 *	\param  dvecA n-dimensional vector with eigenvalues in
 *                    descending order.
 *      \param	c     No. of the considered eigenvalue.
 *      \return       the relative error.
 *      \throw SharkException the type of the exception will be
 *             "size mismatch" and indicates that \em dvecA is not
 *             one-dimensional or that \em amatA or \em vmatA
 *             don't have the same number of rows or columns
 *             as \em dvecA contains number of values
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
 *      none
 *
 *  \par Status
 *      stable
 *
 */
double eigenerr
(
	const Array2D< double >& amatA,
	const Array2D< double >& vmatA,
	const Array  < double >& dvecA,
	unsigned c
)
{
	SIZE_CHECK
	(
		dvecA.ndim() == 1 &&
		dvecA.dim(0) == amatA.dim(0) &&
		dvecA.dim(0) == amatA.dim(1) &&
		dvecA.dim(0) == vmatA.dim(0) &&
		dvecA.dim(0) == vmatA.dim(1)
	)

	unsigned i, j;
	double   s, t;

	s = 0.;
	for (i = 0; i < dvecA.dim(0); i++) {
		t = 0.;
		//multiply matrix amatA with its c's eigenvector
		for (j = 0; j <= i; j++) {
			t += amatA(i, j) * vmatA(j, c);
		}
		//and add product the transpose of amatA and c's eigenvector
		for (; j < dvecA.dim(0); j++) {
			t += amatA(j, i) * vmatA(j, c);
		}
		//subtract righthand side of eigenvalue equation
		t -= vmatA(i, c) * dvecA(c);
		s += t * t;
	}
	//return squared error
	return sqrt(s);
}
