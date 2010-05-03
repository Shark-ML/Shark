//===========================================================================
/*!
 *  \file dlinmin.cpp
 *
 *  \brief Minimizing functions of "N"-dimensional
 *         variables by using derivative information.
 *
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

 *  \par Project:
 *      LinAlg
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of LinAlg. This library is free software;
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
 *
 *
 */
//===========================================================================

#include <cmath>
#include <LinAlg/arrayoptimize.h>
#include <SharkDefs.h>

/*
#ifdef _WIN32
#ifndef __MIN_MAX__
#define __MIN_MAX__
namespace std {
//
// undefine macros min and max to avoid conflicts with template names
//
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

//
// template functions 'min' and 'max' are not defined for _WIN32
// due to name conflicts
//
template < class T > inline T min( T a, T b ) { return a < b ? a : b; }
template < class T > inline T max( T a, T b ) { return a > b ? a : b; }
}
#endif
#endif
*/

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

//===========================================================================
/*!
 *  \brief Minimizes a function of "N" variables by using
 *         derivative information.
 *
 *  Performs a minimization of a function of \f$ N \f$ variables, i.e.
 *  given as input the vectors \f$ P \f$ and \f$ n \f$ and the function
 *  \f$ f \f$, the function finds the scalar \f$ \lambda \f$
 *  that minimizes \f$ f(P + \lambda n) \f$.
 *  \f$ P \f$ is replaced by \f$ P + \lambda n \f$ and \f$ n \f$
 *  by \f$ \lambda n \f$.
 *
 *      \param  p     N-dimensional initial starting point for the
 *                    search, is set to the point where the function
 *                    takes on a minimum.
 *      \param  xi    N-dimensional search direction, is replaced
 *                    by the actual vector displacement that \em p was
 *                    moved.
 *      \param  fret  The function value at the new point \em p.
 *      \param  func  The function that will be minimized.
 *      \param  dfunc The derivation of function \em func.
 *      \return       none.
 *      \throw SharkException the type of the exception will be
 *             "size mismatch" and indicates that \em p is not
 *             one-dimensional or that \em p and \em xi don't have
 *             the same length
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
 *  \sa linmin.cpp
 */
void dlinmin
(
	Array< double >& p,
	const Array< double >& xi,
	double& fret,
	double(*func)(const Array< double >&),
	void(*dfunc)(const Array< double >&, Array< double >&)
)
{
	SIZE_CHECK(p.ndim() == 1 && p.samedim(xi))

	const double   GOLD   = 1.618034;
	const double   GLIMIT = 100.;
	const double   TINY   = 1.0e-20;
	const unsigned ITMAX  = 100;
	const double   ZEPS   = 1.0e-10;
	const double   TOL    = 2.0e-4;

	bool     ok1, ok2;
	unsigned i, iter;
	double   fa, fb, fc, fp, ax, bx, cx;
	double   ulim, dum;
	double   a, b, d(0.), e, fu, fv, fw, fx, q, r, tol1, tol2, u, v, w, x, xm;
	double   du, dv, dw, dx, d1, d2, u1, u2, olde;

	unsigned        n = p.nelem();
	Array< double > xt(n);
	Array< double > df(n);

	//
	// initial guess for the brackets
	//
	ax = 0.;
	bx = 1.;

	//===================================================================

	fa = fp = func(p);

	for (i = 0; i < n; ++i)
		xt(i) = p(i) + xi(i);
	fb = func(xt);

	//ensure that fb <= fa
	if (fb > fa) {
		dum = ax;
		ax = bx;
		bx = dum;
		dum = fb;
		fb = fa;
		fa = dum;
	}

	//evaluate function a golden selection
	cx = bx + GOLD * (bx - ax);
	for (i = 0; i < n; ++i)
		xt(i) = p(i) + xi(i) * cx;
	fc = func(xt);

	while (fb > fc) {//find interval containing the minimum
		r = (bx - ax) * (fb - fc);
		q = (bx - cx) * (fb - fa);
		u = bx - ((bx - cx) * q - (bx - ax) * r) /
			(2. * SIGN(Shark::max(fabs(q - r), TINY), q - r));
		ulim = bx + GLIMIT * (cx - bx);
		if ((bx - u) *(u - cx) > 0.) {//u is in the interval (cx,bx)
			for (i = 0; i < n; ++i)
				xt(i) = p(i) + xi(i) * u;
			fu = func(xt);
			if (fu < fc) {//continue in the interval [bx,u]
				ax = bx;
				bx = u;
				fa = fb;
				fb = fu;
				break;
			}
			else if (fu > fb) {
				cx = u;
				fc = fu;
				break;
			}
			//evalutae new golden selection point
			u = cx + GOLD * (cx - bx);
			for (i = 0; i < n; ++i)
				xt(i) = p(i) + xi(i) * u;
			fu = func(xt);
		}
		else if ((cx - u) *(u - ulim) > 0.) {//u is in the interval [ulim,cx]
			for (i = 0; i < n; ++i)
				xt(i) = p(i) + xi(i) * u;//step in direction given by u
			fu = func(xt);
			if (fu < fc) {//continue in the interval [ax,cx]
				bx = cx;
				cx = u;
				u  = cx + GOLD * (cx - bx);
				fb = fc;
				fc = fu;
				for (i = 0; i < n; ++i)
					xt(i) = p(i) + xi(i) * u;
				fu = func(xt);
			}
		}
		else if ((u - ulim) *(ulim - cx) >= 0.) {//ulim is in the interval [cx,u]
			u = ulim;
			for (i = 0; i < n; ++i)
				xt(i) = p(i) + xi(i) * u;//step in direction given by u
			fu = func(xt);
		}
		else {
			u = cx + GOLD * (cx - bx);
			for (i = 0; i < n; ++i)
				xt(i) = p(i) + xi(i) * u;//step in direction given by u
			fu = func(xt);
		}
		//exchange interval limits and corresponding function values
		ax = bx;
		bx = cx;
		cx = u;
		fa = fb;
		fb = fc;
		fc = fu;
	}

	//=======================================================================

	e = 0.;
	if (ax < cx) {
		a = ax;
		b = cx;
	}
	else {
		a = cx;
		b = ax;
	}

	x = w = v = bx;
	for (i = 0; i < n; ++i)
		xt(i) = p(i) + xi(i) * x;
	fw = fv = fx = func(xt);
	dfunc(xt, df);
	for (dx = 0., i = 0; i < n; ++i)
		dx += xi(i) * df(i);
	dw = dv = dx;

	for (iter = 0; iter < ITMAX; iter++) {
		xm = 0.5 * (a + b);//interval centrum
		tol2 = 2. * (tol1 = TOL * fabs(x) + ZEPS);
		if (fabs(x - xm) <= (tol2 - 0.5 *(b - a))) {
			break;
		}
		// calculate delta x
		if (fabs(e) > tol1) {
			d1 = 2. * (b - a);
			d2 = d1;
			if (dw != dx)
				d1 = (w - x) * dx / (dx - dw);
			if (dv != dx)
				d2 = (v - x) * dx / (dx - dv);
			u1   = x + d1;
			u2   = x + d2;
			ok1  = (a - u1) * (u1 - b) > 0. && dx * d1 <= 0.;
			ok2  = (a - u2) * (u2 - b) > 0. && dx * d2 <= 0.;
			olde = e;
			e    = d;
			if (ok1 || ok2) {
				if (ok1 && ok2)
					d = (fabs(d1) < fabs(d2) ? d1 : d2);
				else if (ok1)
					d = d1;
				else
					d = d2;
				if (fabs(d) <= fabs(0.5 * olde)) {
					u = x + d;
					if (u - a < tol2 || b - u < tol2)
						d = SIGN(tol1, xm - x);
				}
				else {
					d = 0.5 * (e = (dx >= 0. ? a - x : b - x));
				}
			}
			else {
				d = 0.5 * (e = (dx >= 0. ? a - x : b - x));
			}
		}
		else {
			d = 0.5 * (e = (dx >= 0. ? a - x : b - x));
		}
		//update u
		if (fabs(d) >= tol1) {
			u = x + d;
			for (i = 0; i < n; ++i)
				xt(i) = p(i) + xi(i) * u;
			fu = func(xt);
		}
		else {
			u = x + SIGN(tol1, d);
			for (i = 0; i < n; ++i)
				xt(i) = p(i) + xi(i) * u;
			fu = func(xt);
			if (fu > fx)
				break;
		}
		dfunc(xt, df);
		for (du = 0., i = 0; i < n; ++i)
			du += xi(i) * df(i);
		//reduce interval length
		if (fu <= fx) {
			if (u >= x)
				a = x;
			else
				b = x;
			v  = w;
			w  = x;
			x  = u;
			fv = fw;
			fw = fx;
			fx = fu;
			dv = dw;
			dw = dx;
			dx = du;
		}
		else {
			if (u < x)
				a = u;
			else
				b = u;
			if (fu <= fw || w == x) {
				v  = w;
				w  = u;
				fv = fw;
				fw = fu;
				dv = dw;
				dw = du;
			}
			else if (fu < fv || v == x || v == w) {
				v  = u;
				fv = fu;
				dv = du;
			}
		}
	}

	if (iter >= ITMAX)
		std::cerr << "Too many iterations in routine dbrent" << std::endl;

	if (fx < fp) {
		fret = fx;

		for (i = 0; i < n; ++i)
			p(i) += xi(i) * x;
	}
	else
		fret = fp;
}

