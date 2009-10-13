//===========================================================================
/*!
 *  \file eigensymmJacobi2.cpp
 *
 *  \brief Used to calculate the eigenvectors and eigenvalues of
 *         the symmetric matrix "amatA" using a modified Jacobi method.
 *
 *  Here the so-called Jacobi rotation is used to calculate
 *  the eigenvectors and values, but with a modification to
 *  avoid convergence problems.
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
 *  \brief Calculates the eigenvalues and the normalized
 *         eigenvectors of the symmetric matrix "amatA" using a modified
 *         Jacobi method.
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
 *  This function uses the Jacobi method as in #eigensymmJacobi,
 *  but the method is modificated after J. von Neumann to avoid
 *  convergence problems when dealing with low machine precision.
 *
 *      \param  amatA \f$ n \times n \f$ matrix, which must be symmetric, so
 *                    only the bottom
 *                    triangular matrix must contain values.
 *                    Values below the diagonal will be destroyed.
 *      \param	vmatA \f$ n \times n \f$ matrix with the calculated
 *                    normalized
 *                    eigenvectors, each column will contain one
 *                    eigenvector.
 *	\param  dvecA n-dimensional vector with the calculated
 *                    eigenvalues in descending order.
 *      \return       none.
 *      \throw SharkException the type of the exception will be
 *             "size mismatch" and indicates that
 *             	\em amatA is not a square matrix
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
void eigensymmJacobi2
(
	Array2D< double >& amatA,
	Array2D< double >& vmatA,
	Array  < double >& dvecA
)
{
	SIZE_CHECK(amatA.ndim() == 2 && amatA.dim(0) == amatA.dim(1))

	vmatA.resize(amatA, false);
	dvecA.resize(amatA.dim(0), false);

	unsigned n   = dvecA.nelem();

	unsigned ind, i, j, l, m;
	double anorm, thr, aml, all, amm, x, y;
	double *aim, *ail;
	double sinx, sinx2, cosx, cosx2, sincs;

	/*
	 * Diagonalelemente von 'a' nach 'val' kopieren,
	 * 'val' konvergiert gegen die Eigenwerte
	 */
	for (j = 0; j < n; j++) {
		dvecA(j) = amatA(j, j);
	}

	/*
	 * Einheitsmatrix in 'vec' initialisieren
	 * Quadratnorm der Nebendiagonalelemente von 'a' in 'anorm'
	 */
	anorm = 0.0;
	for (i = 0; i < n; i++) {
		for (j = 0; j < i; j++) {
			anorm += amatA(i, j) * amatA(i, j);
			vmatA(i, j) = vmatA(j, i) = 0.0;
		}
		vmatA(i, i) = 1.0;
	}

	if (anorm <= 0.0) {
		goto done; /* Matrix hat keine Nebendiagonalelemente */
	}

	anorm  = sqrt(4 * anorm);
	thr    = anorm;
	anorm /= n;

	while (anorm + thr > anorm)  /* relative Genauigkeit */
		/*  while( 1 + thr > 1 ) */       /* absolute Genauigkeit */
	{
		thr /= n;

		do
		{
			ind = 0;

			for (l = 0; l < n - 1; l++) {
				for (m = l + 1; m < n; m++) {
					if (fabs(aml = amatA(m, l)) < thr) {
						continue;
					}

					ind   = 1;
					all   = dvecA(l);
					amm   = dvecA(m);
					x     = (all - amm) / 2;
					y     = -aml / hypot(aml, x);
					if (x < 0.0) {
						y = -y;
					}
					sinx  = y / sqrt(2 * (1 + sqrt(1 - y * y)));
					sinx2 = sinx * sinx;
					cosx  = sqrt(1 - sinx2);
					cosx2 = cosx * cosx;
					sincs = sinx * cosx;

					/* Spalten l und m rotieren */
					for (i = 0; i < n; i++) {
						if ((i != m) && (i != l)) {
							aim  = i > m ? &amatA(i, m) : &amatA(m, i);
							ail  = i > l ? &amatA(i, l) : &amatA(l, i);
							x    = *ail * cosx - *aim * sinx;
							*aim = *ail * sinx + *aim * cosx;
							*ail = x;
						}
						x = vmatA(i, l);
						y = vmatA(i, m);
						vmatA(i, l) = x * cosx - y * sinx;
						vmatA(i, m) = x * sinx + y * cosx;
					}

					x = 2.0 * aml * sincs;
					dvecA(l) = all * cosx2 + amm * sinx2 - x;
					dvecA(m) = all * sinx2 + amm * cosx2 + x;
					amatA(m, l) = (all - amm) * sincs + aml * (cosx2 - sinx2);
				}
			}
		}
		while (ind);
	}

done:	;

	/*
	 * Eigenwerte sortieren
	 */
	eigensort(vmatA, dvecA);
}
