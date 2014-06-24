//===========================================================================
/*!
 * 
 *
 * \brief      Contains the lapack bindings for the symmetric eigenvalue problem syev.
 *
 * \author      O. Krause
 * \date        2010
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_SYEV_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_SYEV_HPP

#include "../traits.hpp"

namespace shark { namespace blas { namespace bindings {
	
template <typename MatrA, typename VectorB>
void eigensort
(
	matrix_expression<MatrA>& matA,
	vector_expression<VectorB>& eigenValues
){
	SIZE_CHECK(eigenValues().size() == matA().size1());
	SIZE_CHECK(matA().size1() == matA().size2());

	unsigned n = eigenValues().size();

	for (std::size_t i = 0; i < n - 1; i++)
	{
		std::size_t l = arg_max(subrange(eigenValues,i,n))+i;
		if (l != i) {
		        //switch position of i's eigenvalue and the largest remaining eigenvalue
			std::swap(eigenValues()( l ),eigenValues()( i ));
			//switch postions of corresponding eigenvectors
			for (std::size_t j = 0; j < n; j++) {
				std::swap(matA()( j , l ), matA()( j , i ));
			}
		}
	}
}

template <typename MatrA, typename VectorB>
void syev(
	matrix_expression<MatrA>& vmatA,
	vector_expression<VectorB>& dvecA
) {
	SIZE_CHECK(vmatA().size1() == vmatA().size2());
	SIZE_CHECK(vmatA().size1() == dvecA().size());
	
	const unsigned maxIterC = 50;
	unsigned n = vmatA().size1();
	
	blas::vector<double> odvecA(n,0.0);

	unsigned j, k, l, m;
	double   b, c, f, g, h, hh, p, r, s, scale;


	//
	// reduction to tridiagonal form
	//
	for (unsigned i = n; i-- > 1;) {
		h = 0.0;
		scale = 0.0;

		if (i > 1) {
			// scale row
			for (unsigned k = 0; k < i; k++) {
				scale += std::fabs(vmatA()(i, k));
			}
		}

		if (scale == 0.0) {
			odvecA(i) = vmatA()(i, i-1);
		}
		else {
			for (k = 0; k < i; k++) {
				vmatA()(i, k) /= scale;
				h += vmatA()(i, k) * vmatA()(i, k);
			}

			f           = vmatA()(i, i-1);
			g           = f > 0 ? -std::sqrt(h) : std::sqrt(h);
			odvecA(i)       = scale * g;
			h          -= f * g;
			vmatA()(i, i-1) = f - g;
			f           = 0.0;

			for (j = 0; j < i; j++) {
				vmatA()(j, i) = vmatA()(i, j) / (scale * h);
				g = 0.0;

				// form element of a*u
				for (k = 0; k <= j; k++) {
					g += vmatA()(j, k) * vmatA()(i, k);
				}

				for (k = j + 1; k < i; k++) {
					g += vmatA()(k, j) * vmatA()(i, k);
				}

				// form element of p
				f += (odvecA(j) = g / h) * vmatA()(i, j);
			}

			hh = f / (h + h);

			// form reduced a
			for (j = 0; j < i; j++) {
				f     = vmatA()(i, j);
				g     = odvecA(j) - hh * f;
				odvecA(j) = g;

				for (k = 0; k <= j; k++) {
					vmatA()(j, k) -= f * odvecA(k) + g * vmatA()(i, k);
				}
			}

			for (k = i; k--;) {
				vmatA()(i, k) *= scale;
			}
		}

		dvecA()(i) = h;
	}

	dvecA()(0) = odvecA(0) = 0.0;

	// accumulation of transformation matrices
	for (unsigned i = 0; i < n; i++) {
		if (dvecA()(i)) {
			for (j = 0; j < i; j++) {
				g = 0.0;

				for (k = 0; k < i; k++) {
					g += vmatA()(i, k) * vmatA()(k, j);
				}

				for (k = 0; k < i; k++) {
					vmatA()(k, j) -= g * vmatA()(k, i);
				}
			}
		}

		dvecA()(i)     = vmatA()(i, i);
		vmatA()(i, i) = 1.0;

		for (j = 0; j < i; j++) {
			vmatA()(i, j) = vmatA()(j, i) = 0.0;
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
				s = std::fabs(dvecA()(m)) + std::fabs(dvecA()(m+1));
				if (std::fabs(odvecA(m)) + s == s) {
					break;
				}
			}

			p = dvecA()(l);

			if (m != l) {
				if (j++ == maxIterC)
					throw SHARKEXCEPTION("too many iterations in eigendecomposition");

				// form shift
				g = (dvecA()(l+1) - p) / (2.0 * odvecA(l));
				r = std::sqrt(g * g + 1.0);
				g = dvecA()(m) - p + odvecA(l) / (g + ((g) > 0 ? std::fabs(r) : -std::fabs(r)));
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

					g       = dvecA()(i+1) - p;
					r       = (dvecA()(i) - g) * s + 2.0 * c * b;
					p       = s * r;
					dvecA()(i+1) = g + p;
					g       = c * r - b;

					// form vector
					for (k = 0; k < n; k++) {
						f           = vmatA()(k, i+1);
						vmatA()(k, i+1) = s * vmatA()(k, i) + c * f;
						vmatA()(k, i  ) = c * vmatA()(k, i) - s * f;
					}
				}

				dvecA()(l) -= p;
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
	for (unsigned j = n; j--;) {
		s = 0.0;
		for (unsigned i = n; i--;) {
			s += vmatA()(i, j) * vmatA()(i, j);
		}
		s = std::sqrt(s);

		for (unsigned i = n; i--;) {
			vmatA()(i, j) /= s;
		}
	}
}

/** @}*/

}}}

#endif
