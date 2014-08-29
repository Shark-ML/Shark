//===========================================================================
/*!
 *  \file svd.inl
 *
 *  \brief Used for singular value decomposition of rectangular and
 *         square matrices.
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

#ifndef SHARK_LINALG_SVD_INL
#define SHARK_LINALG_SVD_INL

//===========================================================================
/*!
 *  \brief Determines the singular value decomposition of a rectangular
 *  matrix "amatA".
 *
 *  Given a \f$ m \times n \f$ matrix \em amatA, this routine computes its
 *  singular value decomposition, defined as
 *
 *  \f$
 *      A = UWV^T
 *  \f$
 *
 *  where W is an \f$ n \times n \f$ diagonal matrix with positive
 *  or zero elements, the so-called \em singular \em values.
 *  The matrices \em U and \em V are each orthogonal in the sense
 *  that their columns are orthonormal, i.e.
 *
 *  \f$
 *      UU^T = VV^T = V^TV = 1
 *  \f$
 *
 *      \param  amatA The input matrix \em A, with size \f$ m \times n \f$ and
 *                    \f$ m \geq n \f$.
 *      \param  umatA The \f$ m \times n \f$ column-orthogonal matrix \em U
 *                    determined by the function.
 *      \param  vmatA The \f$ n \times n \f$ orthogonal matrix \em V
 *                    determined by the function.
 *      \param  w     n-dimensional vector with the calculated singular values.
 *      \param  maxIterations Number of iterations after which the algorithm gives
 *                    up, if the solution has still not converged.
 *					          Default is 200 Iterations.
 *      \param  ignoreThreshold If set to false, the method throws an exception if
 *              the threshold maxIterations is exceeded. Otherwise it uses the
 *              approximate intermediate results in the further calculations.
 *              The default is true.
 *      \throw  convergence exception, if the solution has not converged after
 *                    maxIterations iterations.
 *      \return       none
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
 */
template<class MatrixT,class MatrixU,class VectorT>
void shark::blas::svd
(
	MatrixT const& amatA,
	MatrixU& umatA,
	MatrixU& vmatA,
	VectorT& w,
	unsigned maxIterations,
	bool ignoreThreshold
)
{
	unsigned m = amatA.size1(); /* rows */
	unsigned n = amatA.size2(); /* cols */

	int flag;
	unsigned i, its, j, jj, k, l, nm(0);
	double anorm, c, f, g, h, s, scale, x, y, z;

	VectorT rv1(n);

	/* copy A to U */
	umatA = amatA;

	/* householder reduction to bidiagonal form */
	g = scale = anorm = 0.0;

	for (i = 0; i < n; i++) {
		l = i + 1;
		rv1(i) = scale * g;
		g = s = scale = 0.0;

		if (i < m) {
			for (k = i; k < m; k++) {
				scale += fabs(umatA(k, i));
			}
			//~ scale += norm_1(column(umatA,i));

			if (scale != 0.0) {
				for (k = i; k < m; k++) {
					umatA(k, i) /= scale;
					s += umatA(k, i) * umatA(k, i);
				}

				f = umatA(i, i);
				g = -copySign(std::sqrt(s), f);
				h = f * g - s;
				umatA(i, i) = f - g;

				for (j = l; j < n; j++) {
					s = 0.0;
					for (k = i; k < m; k++) {
						s += umatA(k, i) * umatA(k, j);
					}

					f = s / h;
					for (k = i; k < m; k++) {
						umatA(k, j) += f * umatA(k, i);
					}
				}

				for (k = i; k < m; k++) {
					umatA(k, i) *= scale;
				}
			}
		}

		w(i) = scale * g;
		g = s = scale = 0.0;

		if (i < m && i != n - 1) {
			for (k = l; k < n; k++) {
				scale += fabs(umatA(i, k));
			}

			if (scale != 0.0) {
				for (k = l; k < n; k++) {
					umatA(i, k) /= scale;
					s += umatA(i, k) * umatA(i, k);
				}

				f = umatA(i, l);
				g = -copySign(std::sqrt(s), f);
				h = f * g - s;
				umatA(i, l) = f - g;

				for (k = l; k < n; k++) {
					rv1(k) = umatA(i, k) / h;
				}

				for (j = l; j < m; j++) {
					s = 0.0;
					for (k = l; k < n; k++) {
						s += umatA(j, k) * umatA(i, k);
					}

					for (k = l; k < n; k++) {
						umatA(j, k) += s * rv1(k);
					}
				}

				for (k = l; k < n; k++) {
					umatA(i, k) *= scale;
				}
			}
		}

		anorm = std::max(anorm, std::abs(w(i)) + std::abs(rv1(i)));
	}

	/* accumulation of right-hand transformations */
	for (l = i = n; i--; l--) {
		if (l < n) {
			if (g != 0.0) {
				for (j = l; j < n; j++) {
					/* double division avoids possible underflow */
					vmatA(j, i) = (umatA(i, j) / umatA(i, l)) / g;
				}

				for (j = l; j < n; j++) {
					s = 0.0;
					for (k = l; k < n; k++) {
						s += umatA(i, k) * vmatA(k, j);
					}

					for (k = l; k < n; k++) {
						vmatA(k, j) += s * vmatA(k, i);
					}
				}
			}

			for (j = l; j < n; j++) {
				vmatA(i, j) = vmatA(j, i) = 0.0;
			}
		}

		vmatA(i, i) = 1.0;
		g = rv1(i);
	}

	/* accumulation of left-hand transformations */
	for (l = i = std::min(m, n); i--; l--) {
		g = w(i);

		for (j = l; j < n; j++) {
			umatA(i, j) = 0.0;
		}

		if (g != 0.0) {
			g = 1.0 / g;

			for (j = l; j < n; j++) {
				s = 0.0;
				for (k = l; k < m; k++) {
					s += umatA(k, i) * umatA(k, j);
				}

				/* double division avoids possible underflow */
				f = (s / umatA(i, i)) * g;

				for (k = i; k < m; k++) {
					umatA(k, j) += f * umatA(k, i);
				}
			}

			for (j = i; j < m; j++) {
				umatA(j, i) *= g;
			}
		}
		else {
			for (j = i; j < m; j++) {
				umatA(j, i) = 0.0;
			}
		}

		umatA(i, i)++;
	}

	/* diagonalization of the bidiagonal form */
	for (k = n; k--;) {
		for (its = 1; its <= maxIterations; its++) {
			flag = 1;

			/* test for splitting */
//			for (l = k + 1; l--; )
			/* Thomas Buecher:
			    Änderung, die fehlerhaft sein kann, aber Absturz verhindert:
				l kann 0 werden --> nm (zwei Zeilen tiefer) wird riesig, da 0-1 auf
				unsigned typ berechnet wird
			*/
			for (l = k; l > 0; l--) {
				/* rv1(0) is always zero, so there is no exit */
				nm = l - 1;

				if (fabs(rv1(l)) + anorm == anorm) {
					flag = 0;
					break;
				}

				if (fabs(w(nm)) + anorm == anorm) {
					break;
				}
			}

			if (flag) {
				/* cancellation of rv1(l) if l greater than 0 */
				c = 0.0;
				s = 1.0;

				for (i = l; i <= k; i++) {
					f = s * rv1(i);
					rv1(i) *= c;

					if (fabs(f) + anorm == anorm) {
						break;
					}

					g = w(i);
					h = hypot(f, g);
					w(i) = h;
					h = 1.0 / h;
					c = g * h;
					s = -f * h;

					for (j = 0; j < m; j++) {
						y = umatA(j, nm);
						z = umatA(j, i);
						umatA(j, nm) = y * c + z * s;
						umatA(j, i ) = z * c - y * s;
					}
				}
			}

			/* test for convergence */
			z = w(k);

			if (l == k) {
				if (z < 0.0) {
					w(k) = -z;
					for (j = 0; j < n; j++) {
						vmatA(j, k) = -vmatA(j, k);
					}
				}
				break;
			}

			if( its == maxIterations ) {
				if(ignoreThreshold)
					break ;
				else
					throw SHARKEXCEPTION("too many iterations for diagonalization of the bidiagonal form during SVD");
			}


			/* shift from bottom 2 by 2 minor */
			x = w(l);
			nm = k - 1;
			y = w(nm);
			g = rv1(nm);
			h = rv1(k);
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = hypot(f, 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + copySign(g, f))) - h)) / x;

			/* next qr transformation */
			c = s = 1.0;

			for (j = l; j < k; j++) {
				i = j + 1;
				g = rv1(i);
				y = w(i);
				h = s * g;
				g *= c;
				z = hypot(f, h);
				rv1(j) = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y *= c;

				for (jj = 0; jj < n; jj++) {
					x = vmatA(jj, j);
					z = vmatA(jj, i);
					vmatA(jj, j) = x * c + z * s;
					vmatA(jj, i) = z * c - x * s;
				}

				z = hypot(f, h);
				w(j) = z;

				/* rotation can be arbitrary if z is zero */
				if (z != 0.0) {
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}

				f = c * g + s * y;
				x = c * y - s * g;

				for (jj = 0; jj < m; jj++) {
					y = umatA(jj, j);
					z = umatA(jj, i);
					umatA(jj, j) = y * c + z * s;
					umatA(jj, i) = z * c - y * s;
				}
			}

			rv1(l) = 0.0;
			rv1(k) = f;
			w(k) = x;
		}
	}
}
#endif
