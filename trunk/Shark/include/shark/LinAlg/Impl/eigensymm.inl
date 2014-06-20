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
	SIZE_CHECK(amatA.size1() == amatA.size2());

	unsigned n = amatA.size1();

	vmatA.resize(n,n);
	dvecA.resize(n);
	odvecA.resize(n);
	vmatA.clear();
	dvecA.clear();
	odvecA.clear();

	const unsigned maxIterC = 50;



	unsigned j, k, l, m;
	double   b, c, f, g, h, hh, p, r, s, scale;

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

	//
	// reduction to tridiagonal form
	//
	for (unsigned i = n; i-- > 1;) {
		h = 0.0;
		scale = 0.0;

		if (i > 1) {
			// scale row
			for (unsigned k = 0; k < i; k++) {
				scale += std::fabs(vmatA(i, k));
			}
		}

		if (scale == 0.0) {
			odvecA(i) = vmatA(i, i-1);
		}
		else {
			for (k = 0; k < i; k++) {
				vmatA(i, k) /= scale;
				h += vmatA(i, k) * vmatA(i, k);
			}

			f           = vmatA(i, i-1);
			g           = f > 0 ? -std::sqrt(h) : std::sqrt(h);
			odvecA(i)       = scale * g;
			h          -= f * g;
			vmatA(i, i-1) = f - g;
			f           = 0.0;

			for (j = 0; j < i; j++) {
				vmatA(j, i) = vmatA(i, j) / (scale * h);
				g = 0.0;

				// form element of a*u
				for (k = 0; k <= j; k++) {
					g += vmatA(j, k) * vmatA(i, k);
				}

				for (k = j + 1; k < i; k++) {
					g += vmatA(k, j) * vmatA(i, k);
				}

				// form element of p
				f += (odvecA(j) = g / h) * vmatA(i, j);
			}

			hh = f / (h + h);

			// form reduced a
			for (j = 0; j < i; j++) {
				f     = vmatA(i, j);
				g     = odvecA(j) - hh * f;
				odvecA(j) = g;

				for (k = 0; k <= j; k++) {
					vmatA(j, k) -= f * odvecA(k) + g * vmatA(i, k);
				}
			}

			for (k = i; k--;) {
				vmatA(i, k) *= scale;
			}
		}

		dvecA(i) = h;
	}

	dvecA(0) = odvecA(0) = 0.0;

	// accumulation of transformation matrices
	for (unsigned i = 0; i < n; i++) {
		if (dvecA(i)) {
			for (j = 0; j < i; j++) {
				g = 0.0;

				for (k = 0; k < i; k++) {
					g += vmatA(i, k) * vmatA(k, j);
				}

				for (k = 0; k < i; k++) {
					vmatA(k, j) -= g * vmatA(k, i);
				}
			}
		}

		dvecA(i)     = vmatA(i, i);
		vmatA(i, i) = 1.0;

		for (j = 0; j < i; j++) {
			vmatA(i, j) = vmatA(j, i) = 0.0;
		}
	}

	//
	// eigenvalues from tridiagonal form
	//
	if (n <= 1) {
		return;
	}

	for (unsigned i = 1; i < n; i++) {
		odvecA(i-1) = odvecA(i);
	}

	odvecA(n-1) = 0.0;

	for (l = 0; l < n; l++) {
		j = 0;

		do {
			// look for small sub-diagonal element
			for (m = l; m < n - 1; m++) {
				s = std::fabs(dvecA(m)) + std::fabs(dvecA(m+1));
				if (std::fabs(odvecA(m)) + s == s) {
					break;
				}
			}

			p = dvecA(l);

			if (m != l) {
				if (j++ == maxIterC)
					throw SHARKEXCEPTION("too many iterations in eigendecomposition");

				// form shift
				g = (dvecA(l+1) - p) / (2.0 * odvecA(l));
				r = std::sqrt(g * g + 1.0);
				g = dvecA(m) - p + odvecA(l) / (g + ((g) > 0 ? std::fabs(r) : -std::fabs(r)));
				s = c = 1.0;
				p = 0.0;

				for (unsigned i = m; i-- > l;) {
					f = s * odvecA(i);
					b = c * odvecA(i);

					if (std::fabs(f) >= std::fabs(g)) {
						c       = g / f;
						r       = std::sqrt(c * c + 1.0);
						odvecA(i+1) = f * r;
						s       = 1.0 / r;
						c      *= s;
					}
					else {
						s       = f / g;
						r       = std::sqrt(s * s + 1.0);
						odvecA(i+1) = g * r;
						c       = 1.0 / r;
						s      *= c;
					}

					g       = dvecA(i+1) - p;
					r       = (dvecA(i) - g) * s + 2.0 * c * b;
					p       = s * r;
					dvecA(i+1) = g + p;
					g       = c * r - b;

					// form vector
					for (k = 0; k < n; k++) {
						f           = vmatA(k, i+1);
						vmatA(k, i+1) = s * vmatA(k, i) + c * f;
						vmatA(k, i  ) = c * vmatA(k, i) - s * f;
					}
				}

				dvecA(l) -= p;
				odvecA(l)  = g;
				odvecA(m)  = 0.0;
			}
		}
		while (m != l);
	}

	//
	// sorting eigenvalues
	//
	eigensort(vmatA, dvecA);

	//
	// normalizing eigenvectors
	//
	for (unsigned j = n-1; j != 0; --j) {
		s = 0.0;
		for (unsigned i = n-1; i != 0; --i) {
			s += vmatA(i, j) * vmatA(i, j);
		}
		s = std::sqrt(s);

		for (unsigned i = n-1; i != 0; --i) {
			vmatA(i, j) /= s;
		}
	}
}
#endif
