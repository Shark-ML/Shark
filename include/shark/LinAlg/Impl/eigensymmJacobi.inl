//===========================================================================
/*!
 *  \brief Used to calculate the eigenvectors and eigenvalues of
 *         the symmetric matrix "amatA".
 *
 *  Here the so-called Jacobi rotation is used to calculate
 *  the eigenvectors and values.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Copyright (c) 1998-2000:
 *      Institut fuer Neuroinformatik<BR>
 *      Ruhr-Universitaet Bochum<BR>
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

#ifndef SHARK_LINALG_EIGENSYMM_JACOBI_INL
#define SHARK_LINALG_EIGENSYMM_JACOBI_INL

//===========================================================================
/*!
 *  \brief Calculates the eigenvalues and the normalized
 *         eigenvectors of the symmetric matrix "amatA" using the Jacobi
 *         method.
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
 *  For the calculation of the eigenvectors and eigenvalues the
 *  so called Jacobi rotation is used to annihilate one of the
 *  off-diagonal elements with the basic Jacobi rotation
 *  given as matrix of the form
 *
 *  \f$
 *      P_{pq} =
 *      \left(
 *      \begin{array}{*{7}{c}}
 *          1                                                \\
 *             & \dots                                       \\
 *             &       & c      & \dots & s                  \\
 *             &       & \vdots & 1     & \vdots             \\
 *             &       & -s     & \dots & c                  \\
 *             &       &        &       &        & \dots     \\
 *             &       &        &       &        &       & 1 \\
 *      \end{array}
 *      \right)
 *  \f$
 *
 *  In this matrix all the diagonal elements are unity except for the
 *  two elemnts \em c in rows (and columns) \em p and \em q. All
 *  off-diagonal elements are zero except the two elements \em s
 *  and - \em s. The numbers \em c and \em s are the cosine of a
 *  rotation angle \f$ \Phi \f$, so \f$ c^2 + s^2 = 1\f$.
 *  Successive rotations lead to the off-diagonal elements
 *  getting smaller and smaller, until the matrix is diagonal to
 *  machine precision. Accumulating the product of the transformations
 *  as you go gives the matrix of eigenvectors, while the elements
 *  of the final diagonal matrix are the eigenvalues.
 *  Use this function for the calculation of the eigenvalues and
 *  eigenvectors for matrices \em amatA with moderate order
 *  not greater than 10.
 *
 *      \param  amatA \f$ n \times n \f$ matrix, which must be symmetric, so
 *                    only the upper
 *                    triangular matrix must contain values.
 *                    Values above the diagonal will be destroyed.
 *      \param	vmatA \f$ n \times n \f$ matrix with the calculated
 *                    normalized
 *                    eigenvectors, each column will contain an
 *                    eigenvector.
 *	\param  dvecA n-dimensional vector with the calculated
 *                    eigenvalues in descending order.
 *      \return       none.
 *      \throw SharkException
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
template<class MatrixT,class MatrixU,class VectorT>
void shark::eigensymmJacobi
(
	MatrixT& amatA,
	MatrixU& vmatA,
	VectorT& dvecA
)
{
	SIZE_CHECK(amatA.size1() == amatA.size2());

	const unsigned maxIterC = 50;
	size_t n = amatA.size1();

	vmatA.resize(n,n, false);
	dvecA.resize(n, false);

	unsigned j, iq, ip, i;
	double thresh, theta, tau, t, sm, s, h, g, c;

	VectorT b(n);
	VectorT z(n);

	for (ip = 0; ip < n; ip++) {
		for (iq = 0; iq < n; iq++) {
			vmatA(ip, iq) = 0.0;
		}
		vmatA(ip, ip) = 1.0;
		b(ip) = dvecA(ip) = amatA(ip, ip);
		z(ip) = 0.0;
	}

	for (i = 1; i <= maxIterC; i++) {
		sm = 0.0;

		for (ip = 0; ip < n - 1; ip++) {
			for (iq = ip + 1; iq < n; iq++) {
				sm += std::fabs(amatA(ip, iq));
			}
		}

		if (sm == 0.0) {
			eigensort(vmatA, dvecA);

			return;
		}

		thresh = i < 4 ? 0.2 * sm / (n * n) : 0.0;

		for (ip = 0; ip < n - 1; ip++) {
			for (iq = ip + 1; iq < n; iq++) {
				g = 100.0 * std::fabs(amatA(ip, iq));

				if (i > 4 && std::fabs(dvecA(ip)) + g == std::fabs(dvecA(ip))
						&& std::fabs(dvecA(iq)) + g == std::fabs(dvecA(iq))) {
					amatA(ip, iq) = 0.0;
				}
				else if (std::fabs(amatA(ip, iq)) > thresh) {
					h = dvecA(iq) - dvecA(ip);

					if (std::fabs(h) + g == std::fabs(h)) {
						t = (amatA(ip, iq)) / h;
					}
					else {
						theta = 0.5 * h / (amatA(ip, iq));
						t = 1.0 / (std::fabs(theta) + std::sqrt(1. + theta * theta));
						if (theta < 0.0) {
							t = -t;
						}
					}

					c   = 1.0 / std::sqrt(1 + t * t);
					s   = t * c;
					tau = s / (1.0 + c);
					h   = t * amatA(ip, iq);
					z(ip) -= h;
					z(iq) += h;
					dvecA(ip) -= h;
					dvecA(iq) += h;
					amatA(ip, iq) = 0.0;

					for (j = 0; j < ip; j++) {
						g = amatA(j, ip);
						h = amatA(j, iq);
						amatA(j, ip) = g - s * (h + g * tau);
						amatA(j, iq) = h + s * (g - h * tau);

					}
					for (j = ip + 1; j < iq; j++) {
						g = amatA(ip, j);
						h = amatA(j, iq);
						amatA(ip, j) = g - s * (h + g * tau);
						amatA(j, iq) = h + s * (g - h * tau);
					}
					for (j = iq + 1; j < n; j++) {
						g = amatA(ip, j);
						h = amatA(iq, j);
						amatA(ip, j) = g - s * (h + g * tau);
						amatA(iq, j) = h + s * (g - h * tau);
					}
					for (j = 0; j < n; j++) {
						g = vmatA(j, ip);
						h = vmatA(j, iq);
						vmatA(j, ip) = g - s * (h + g * tau);
						vmatA(j, iq) = h + s * (g - h * tau);
					}
				}
			}
		}

		for (ip = 0; ip < n; ip++) {
			b(ip) += z(ip);
			dvecA(ip)  = b(ip);
			z(ip)  = 0.0;
		}
	}

	throw SHARKEXCEPTION("too many iterations in eigendecomposition");
}
#endif
