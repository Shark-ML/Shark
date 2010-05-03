//===========================================================================
/*!
 *  \file BFGS.cpp
 *
 *  \brief Offers the Broyden-Fletcher-Goldfarb-Shanno algorithm
 *         for the optimization of models.
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Copyright (c) 1999
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      ReClaM
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of ReClaM. This library is free software;
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
 * ---------------------------------------------------------------------
 *
 * This code is a modification of the original code from the
 * library:
 *
 * ============================================================
 * COOOL           version 1.1           ---     Nov,  1995
 * Center for Wave Phenomena, Colorado School of Mines
 * ============================================================
 *
 * This code is part of a preliminary release of COOOL (CWP
 * Object-Oriented Optimization Library) and associated class
 * libraries.
 *
 * The COOOL library is a free software. You can do anything you want
 * with it including make a fortune.  However, neither the authors,
 * the Center for Wave Phenomena, nor anyone else you can think of
 * makes any guarantees about anything in this package or any aspect
 * of its functionality.
 *
 * Since you've got the source code you can also modify the
 * library to suit your own purposes. We would appreciate it
 * if the headers that identify the authors are kept in the
 * source code.
 *
 * ==================================
 * author:  H. Lydia Deng, 06/17/96
 * ==================================
 *
 */
//===========================================================================


#include <cmath>
#include <iostream>
#include <float.h>
#include <SharkDefs.h>
#include <ReClaM/BFGS.h>


#define DLINMIN


//===========================================================================
/*!
 *  \brief Updates the weights of the network by using
 *         the BFGS optimization algorithm.
 *
 *  The given input patterns \em in and their corresponding target
 *  values \em target are used to calculate the error of the currently
 *  used model. The resulting error function is then minimized by the
 *  BFGS algorithm, the weights of the network are updated. <br>
 *
 *      \param in         Input patterns for the currently used model.
 *      \param target     Target values corresponding to the input patterns.
 *      \param gtol       Convergence requirement on zeoring the gradient.
 *      \param iter       Number of iterations that were
 *                        performed by the function.
 *      \param fret       Minimum value of the error function.
 *      \param iterMax    Max. number of allowed iterations.
 *      \return none
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      Unstable
 *
 */

double BFGS::optimize(
	Model& model,
	ErrorFunction& errorfunction,
	const Array<double>& input,
	const Array<double>& target)
{
	bfgs(model, errorfunction, input, target, FRET);
	return FRET;
}

double BFGS::optimize(
	Model& model,
	ErrorFunction& errorfunction,
	const Array<double>& input,
	const Array<double>& target,
	double gtol,
	int iterMax)
{
	unsigned int iter;
	bfgs(model, errorfunction, input, target, gtol, iter, FRET, iterMax);
	return FRET;
}


void BFGS::bfgs(
	Model& model,
	ErrorFunction& errorfunction,
	const Array<double> &in,
	const Array<double> &target,
	double gtol,
	unsigned& iter,
	double&   fret,
	unsigned iterMax)
{
	unsigned i, j, n = BFGS_n;
	DEDW.resize(n, false);
//    part of Init

	if (firstIteration)
	{
		fret = errorfunction.errorDerivative(model, in, target, DEDW);
		BFGS_g0 = DEDW;
		firstIteration = false;
	}
	Array<double> p(n);
	for (i = 0;i < n;i++)p(i) = model.getParameter(i);

	double   d = 0., err, scale;
	double   test, temp;

	Array< double > p1(n);
	Array< double > g0(n);
	Array< double > g1(n);
	Array< double > s(n);

	Array< double > H(n, n);

	Array< double > gamma(n);
	Array< double > delta(n);

	iter = 0;

	double r = errorfunction.errorDerivative(model, in, target, DEDW);
	g0 = DEDW;

	for (err = 0., i = 0; i < n; ++i)
		err += g0(i) * g0(i);
	if (err < gtol*gtol)
	{
		fret = r;
		for (i = 0;i < n;i++)model.setParameter(i, p(i));
		return;
	}

	for (i = 0; i < n; ++i)
	{
		for (j = 0; j < n; ++j)
			H(i, j) = 0.;
		H(i, i) = 1.;
	}

	for (; ;)
	{
		for (i = 0; i < n; ++i)
			for (s(i) = 0., j = 0; j < n; ++j)
				s(i) -= H(i, j) * g0(j);

		//
		// line search
		//


		switch (lineSearchType)
		{
		case Dlinmin:
//  dlinmin( p1, s, fret, in, target);
			dlinmin(model, errorfunction, DEDW, p1, s, fret, in, target);
			break;
		case Linmin:
//	  linmin( p1, s, fret, in, target);
			linmin(model, errorfunction, DEDW, p1, s, fret, in, target);
			break;
		case Cblnsrch:
//  cblnsrch(model, errorfunction, CG_fret, dedw, CG_xi, CG_fret, input, target);
			cblnsrch(model, errorfunction, fret, DEDW, BFGS_p1, fret, in, target);
//  cblnsrch( BFGS_p, errorfuncion.errorDerrivative(input,target,DEDW), BFGS_g0, BFGS_s, BFGS_p1, fret, in, target);
			break;
		}

		for (i = 0;i < n;i++)model.setParameter(i, p1(i));
		errorfunction.errorDerivative(model, in, target, DEDW);
		g1 = DEDW;
		for (err = 0., i = 0; i < n; ++i)
			err += g1(i) * g1(i);
		if (++iter >= iterMax || err <= gtol*gtol)
			break;

		//
		// test for convergence
		//
		test = 0.;
		for (i = 0; i < n; i++)
		{
			temp = fabs(p1(i) - p(i)) / Shark::max((p(i)), 1.);
			if (temp > test)
				test = temp;
		}
		if (test < gtol)
		{
			return;
		}

		for (d = 0., i = 0; i < n; ++i)
			d += (gamma(i) = g1(i) - g0(i)) * (delta(i) = p1(i) - p(i));

		for (i = 0; i < n; ++i)
			for (s(i) = 0., j = 0; j < n; ++j)
				s(i) += H(i, j) * gamma(j);

		if (d < 1e-8)
		{
			for (i = 0; i < n; ++i)
			{
				for (j = 0; j < n; ++j)
					H(i, j) = 0.;
				H(i, i) = 1.;
			}
		}
		else
		{
			for (scale = 0., i = 0; i < n; ++i)
				scale += gamma(i) * s(i);
			scale = (scale / d + 1) / d;

			for (i = 0; i < n; ++i)
			{
				g0(i) = g1(i);
				p(i) = p1(i);
				for (j = 0; j < n; ++j)
					H(i, j) += scale * delta(i) * delta(j)
							   - (s(i) * delta(j) + s(j) * delta(i)) / d;
			}
		}
	}

	for (i = 0; i < n; ++i)model.setParameter(i, p1(i));

}


//===========================================================================
/*!
 *  \brief Initializes some internal variables used by the BFGS algorithm,
 *         calculates the current error of the model.
 *
 *  The given parameters are used to initialize some variables
 *  used by the BFGS and the line search algorithms and to allocate
 *  memory for internal structures. The current error value of the
 *  used model is determined.
 *
 *      \param in         Input patterns for the currently used model.
 *      \param target     Target values corresponding to the input patterns.
 *      \param fret       Initial error of the currently used model.
 *      \param ls         Type of LineSearch algorithm, that is used
 *                        by the BFGS algorithm. Possible values:
 *                        "Dlinmin" (default value), "Linmin", "Cblnsrch"
 *      \param ax         Initial left bracket for the line search
 *                        algorithms, the default bracket is "0".
 *      \param bx         Initial right bracket for the line search
 *                        algorithms, the default bracket is "1".
 *      \param lambda     The initial value for the searched \f$\lambda\f$,
 *                        that will minimize the function used by the
 *                        line search algorithm. The default value is "0.25".
 *      \return none
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
unsigned i;
void BFGS::init(Model& model)
{
	initBfgs(model);
}


void BFGS::initBfgs(Model& model, LineSearch ls,
					double ax, double bx, double lambda)
{
	firstIteration = true;
	BFGS_n = model.getParameterDimension();
	unsigned i, j;
	BFGS_p.resize(BFGS_n, false);
	for (i = 0;i < BFGS_n;i++)BFGS_p(i) = model.getParameter(i);
	lineSearchType = ls;
	LS_lambda      = lambda;
	LS_ax          = ax;
	LS_bx          = bx;




	H.resize(BFGS_n, BFGS_n, false);
	BFGS_p1.resize(BFGS_n, false);
	BFGS_g0.resize(BFGS_n, false);
	BFGS_g1.resize(BFGS_n, false);
	BFGS_s.resize(BFGS_n, false);

	BFGS_gamma.resize(BFGS_n, false);
	BFGS_delta.resize(BFGS_n, false);

	for (i = 0;i < BFGS_n;i++)model.setParameter(i, BFGS_p(i));
	for (i = 0; i < BFGS_n; ++i)
	{
		for (j = 0; j < BFGS_n; ++j) H(i, j) = 0.;
		H(i, i) = 1.;
	}
}

void BFGS::reset()
{
	unsigned i, j;
	BFGS_g0 = DEDW;
	BFGS_p1 = 0.; BFGS_g1 = 0.; BFGS_s = 0.; BFGS_gamma = 0.; BFGS_delta = 0.;
	for (i = 0; i < BFGS_n; ++i)
	{
		for (j = 0; j < BFGS_n; ++j)
			H(i, j) = 0.;
		H(i, i) = 1.;
	}
}


//===========================================================================
/*!
 *  \brief Updates the weights of the network by using
 *         the BFGS optimization algorithm, that was previously initialized
 *         and returns the minimum value by reference.
 *
 *  The given input patterns \em in and their corresponding target
 *  values \em target are used to calculate the error of the currently
 *  used model. The resulting error function is then minimized by the
 *  BFGS algorithm, the weights of the network are updated. <br>
 *  This method needs far less parameters than the other BFGS method,
 *  because it is guessed, that one of the #initBfgs methods was
 *  called before.
 *
 *      \param in         Input patterns for the currently used model.
 *      \param target     Target values corresponding to the input patterns.
 *      \param fret       Minimum value of the error function.
 *      \return none
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void BFGS::bfgs(Model& model,
				ErrorFunction& errorfunction,
				const Array<double> &in,
				const Array<double> &target,
				double& fret)
{
	unsigned i, j;
	DEDW.resize(BFGS_n, false);
	//    part of Init

	if (firstIteration)
	{
		fret = errorfunction.errorDerivative(model, in, target, DEDW);
		BFGS_g0 = DEDW;
		firstIteration = false;
	}
	for (i = 0; i < BFGS_n; ++i)
	{
		for (BFGS_s(i) = 0., j = 0; j < BFGS_n; ++j)
			BFGS_s(i) -= H(i, j) * BFGS_g0(j);
	}

	//
	// line search
	//

	BFGS_p1 = BFGS_p;

	switch (lineSearchType)
	{
	case Dlinmin:
//  dlinmin( p1, s, fret, in, target);
		dlinmin(model, errorfunction, DEDW, BFGS_p1, BFGS_s, fret, in, target);
		break;
	case Linmin:
//	  linmin( p1, s, fret, in, target);
		linmin(model, errorfunction, DEDW, BFGS_p1, BFGS_s, fret, in, target);
		break;
	case Cblnsrch:
//  cblnsrch(model, errorfunction, CG_fret, dedw, CG_xi, CG_fret, input, target);
		cblnsrch(model, errorfunction, fret, DEDW, BFGS_p1, fret, in, target);
//  cblnsrch( BFGS_p, errorfuncion.errorDerrivative(input,target,DEDW), BFGS_g0, BFGS_s, BFGS_p1, fret, in, target);
		break;
	}

	for (i = 0;i < BFGS_n;i++)model.setParameter(i, BFGS_p1(i));

	errorfunction.errorDerivative(model, in, target, DEDW);
	BFGS_g1 = DEDW;

	for (BFGS_d = 0., i = 0; i < BFGS_n; ++i)
		BFGS_d += (BFGS_gamma(i) = BFGS_g1(i) - BFGS_g0(i)) * (BFGS_delta(i) = BFGS_p1(i) - BFGS_p(i));

	for (i = 0; i < BFGS_n; ++i)
		for (BFGS_s(i) = 0., j = 0; j < BFGS_n; ++j)
			BFGS_s(i) += H(i, j) * BFGS_gamma(j);

	if (BFGS_d < 1e-8)
	{
		for (i = 0; i < BFGS_n; ++i)
		{
			for (j = 0; j < BFGS_n; ++j)
				H(i, j) = 0.;
			H(i, i) = 1.;
		}
	}
	else
	{
		for (BFGS_scale = 0., i = 0; i < BFGS_n; ++i)
			BFGS_scale += BFGS_gamma(i) * BFGS_s(i);
		BFGS_scale = (BFGS_scale / BFGS_d + 1) / BFGS_d;

		for (i = 0; i < BFGS_n; ++i)
		{
			BFGS_g0(i) = BFGS_g1(i);
			BFGS_p(i) = BFGS_p1(i);
			for (j = 0; j < BFGS_n; ++j)
				H(i, j) += BFGS_scale * BFGS_delta(i) * BFGS_delta(j)
						   - (BFGS_s(i) * BFGS_delta(j) + BFGS_s(j) * BFGS_delta(i)) / BFGS_d;
		}
	}

	for (i = 0; i < BFGS_n; ++i)
		BFGS_p(i) = BFGS_p1(i);

	for (i = 0;i < BFGS_n;i++)model.setParameter(i, BFGS_p(i));
}


//! Just used to override the standard definition.
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))



void BFGS::dlinmin(
	Model& model,
	ErrorFunction& errorfunction,
	Array<double>& dedw,
	Array< double >& p,
	const Array< double >& xi,
	double& fret,
	const Array< double >&in,
	const Array< double >&target)
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
	double   a, b, d = 0, e, fu, fv, fw, fx, q, r, tol1, tol2, u, v, _w, x, xm;
	double   du, dv, dw, dx, d1, d2, u1, u2, olde;



	unsigned        n = BFGS_n;
	Array< double > xt(n);
	Array< double > generalDerivative(n);

	ax = LS_ax;
	bx = LS_bx;

	//===================================================================

	for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*ax);
	fp = fa = errorfunction.error(model, in, target);

	for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*bx);
	fb = errorfunction.error(model, in, target);

	if (fb > fa)
	{

		dum = ax;
		ax = bx;
		bx = dum;
		dum = fb;
		fb = fa;
		fa = dum;
	}

	cx = bx + GOLD * (bx - ax);

	for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*cx);
	fc = errorfunction.error(model, in, target);

	while (fb > fc)
	{
		r = (bx - ax) * (fb - fc);
		q = (bx - cx) * (fb - fa);
		u = bx - ((bx - cx) * q - (bx - ax) * r) /
			(2. * SIGN(Shark::max(fabs(q - r), TINY), q - r));
		ulim = bx + GLIMIT * (cx - bx);
		if ((bx - u) *(u - cx) > 0.)
		{
			for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);

			fu = errorfunction.error(model, in, target);

			if (fu < fc)
			{
				ax = bx;
				bx = u;
				fa = fb;
				fb = fu;
				break;
			}
			else if (fu > fb)
			{
				cx = u;
				fc = fu;
				break;
			}
			u = cx + GOLD * (cx - bx);

			for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);
			fu = errorfunction.error(model, in, target);

		}
		else if ((cx - u) *(u - ulim) > 0.)
		{
			for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);
			fu = errorfunction.error(model, in, target);

			if (fu < fc)
			{
				bx = cx;
				cx = u;
				u  = cx + GOLD * (cx - bx);
				fb = fc;
				fc = fu;
				for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);

				fu = errorfunction.error(model, in, target);
			}
		}
		else if ((u - ulim) *(ulim - cx) >= 0.)
		{
			u = ulim;
			for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);
			fu = errorfunction.error(model, in, target);
		}
		else
		{
			u = cx + GOLD * (cx - bx);
			for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);
			fu = errorfunction.error(model, in, target);
		}
		ax = bx;
		bx = cx;
		cx = u;
		fa = fb;
		fb = fc;
		fc = fu;
	}
	//=======================================================================

	e = 0.;
	if (ax < cx)
	{
		a = ax;
		b = cx;
	}
	else
	{
		a = cx;
		b = ax;
	}
	x = _w = v = bx;
	for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*x);
	fw = fv = fx = errorfunction.errorDerivative(model, in, target, dedw);
	generalDerivative = dedw;

	for (dx = 0., i = 0; i < n; ++i)
		dx += xi(i) * generalDerivative(i);
	dw = dv = dx;

	for (iter = 0; iter < ITMAX; iter++)
	{
		xm = 0.5 * (a + b);
		tol2 = 2. * (tol1 = TOL * fabs(x) + ZEPS);
		if (fabs(x - xm) <= (tol2 - 0.5 *(b - a)))
		{
			break;
		}
		if (fabs(e) > tol1)
		{
			d1 = 2. * (b - a);
			d2 = d1;
			if (dw != dx)
				d1 = (_w - x) * dx / (dx - dw);
			if (dv != dx)
				d2 = (v - x) * dx / (dx - dv);
			u1   = x + d1;
			u2   = x + d2;
			ok1  = (a - u1) * (u1 - b) > 0. && dx * d1 <= 0.;
			ok2  = (a - u2) * (u2 - b) > 0. && dx * d2 <= 0.;
			olde = e;
			e    = d;
			if (ok1 || ok2)
			{
				if (ok1 && ok2)
					d = (fabs(d1) < fabs(d2) ? d1 : d2);
				else if (ok1)
					d = d1;
				else
					d = d2;
				if (fabs(d) <= fabs(0.5 * olde))
				{
					u = x + d;
					if (u - a < tol2 || b - u < tol2)
						d = SIGN(tol1, xm - x);
				}
				else
				{
					d = 0.5 * (e = (dx >= 0. ? a - x : b - x));
				}
			}
			else
			{
				d = 0.5 * (e = (dx >= 0. ? a - x : b - x));
			}
		}
		else
		{
			d = 0.5 * (e = (dx >= 0. ? a - x : b - x));
		}
		if (fabs(d) >= tol1)
		{
			u = x + d;
			for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);
			fu = errorfunction.errorDerivative(model, in, target, dedw);
			generalDerivative = dedw;
		}
		else
		{
			u = x + SIGN(tol1, d);
			for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);
			fu = errorfunction.errorDerivative(model, in, target, dedw);
			generalDerivative = dedw;
			if (fu > fx)
				break;
		}
		for (du = 0., i = 0; i < n; ++i)
			du += xi(i) * generalDerivative(i);
		if (fu <= fx)
		{
			if (u >= x)
				a = x;
			else
				b = x;
			v  = _w;
			_w  = x;
			x  = u;
			fv = fw;
			fw = fx;
			fx = fu;
			dv = dw;
			dw = dx;
			dx = du;
		}
		else
		{
			if (u < x)
				a = u;
			else
				b = u;
			if (fu <= fw || _w == x)
			{
				v  = _w;
				_w  = u;
				fv = fw;
				fw = fu;
				dv = dw;
				dw = du;
			}
			else if (fu < fv || v == x || v == _w)
			{
				v  = u;
				fv = fu;
				dv = du;
			}
		}
	}
	if (iter >= ITMAX)
		throw SHARKEXCEPTION("BFGS: too many iterations in routine dbrent");
	if (fx < fp)
	{
		fret = fx;
		for (i = 0; i < n; ++i)
			p(i) += xi(i) * x;
	}
	else
		fret = fp;
}

// linmin.cpp

void BFGS::linmin(
	Model& model,
	ErrorFunction& errorfunction,
	Array<double>& dedw,
	Array< double >& p,
	const Array< double >& xi,
	double& fret,
	const Array< double >&in,
	const Array< double >&target)
{

	SIZE_CHECK(p.ndim() == 1 && p.samedim(xi))
	const double   GOLD   = 1.618034;
	const double   GLIMIT = 100.;
	const double   TINY   = 1.0e-20;
	const unsigned ITMAX  = 100;
	const double   CGOLD  = 0.3819660;
	const double   ZEPS   = 1.0e-10;
	const double   TOL    = 2.0e-4;

	unsigned i, iter;
	double   fa, fb, fc, ax, bx, cx;
	double   ulim, dum;
	double   a, b, d = 0., e, etemp, fu, fv, fw, fx, o, q, r, tol1, tol2, u, v, _w, x, xm;
	unsigned        n = BFGS_n;
	Array< double > xt(n);

	ax = LS_ax;
	bx = LS_bx;

	//===================================================================

	for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*ax);
	double initial = fa = errorfunction.error(model, in, target);

	for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*bx);
	fb = errorfunction.error(model, in, target);

	if (fb > fa)
	{
		dum = ax;
		ax  = bx;
		bx  = dum;
		dum = fb;
		fb  = fa;
		fa  = dum;
	}
	cx = bx + GOLD * (bx - ax);
	for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*cx);
	fc = errorfunction.error(model, in, target);
	while (fb > fc)
	{
		r = (bx - ax) * (fb - fc);
		q = (bx - cx) * (fb - fa);
		u = bx - ((bx - cx) * q - (bx - ax) * r) /
			(2. * SIGN(Shark::max(fabs(q - r), TINY), q - r));
		ulim = bx + GLIMIT * (cx - bx);
		if ((bx - u) *(u - cx) > 0.)
		{
			for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);
			fu = errorfunction.error(model, in, target);
			if (fu < fc)
			{
				ax = bx;
				bx = u;
				fa = fb;
				fb = fu;
				break;
			}
			else if (fu > fb)
			{
				cx = u;
				fc = fu;
				break;
			}
			u = cx + GOLD * (cx - bx);
			for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);
			fu = errorfunction.error(model, in, target);
		}
		else if ((cx - u) *(u - ulim) > 0.)
		{
			for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);
			fu = errorfunction.error(model, in, target);
			if (fu < fc)
			{
				bx = cx;
				cx = u;
				u  = cx + GOLD * (cx - bx);
				fb = fc;
				fc = fu;
				for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);
				fu = errorfunction.error(model, in, target);
			}
		}
		else if ((u - ulim) *(ulim - cx) >= 0.)
		{
			u = ulim;
			for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);
			fu = errorfunction.error(model, in, target);
		}
		else
		{
			u = cx + GOLD * (cx - bx);
			for (i = 0; i < n; ++i)  model.setParameter(i, p(i) + xi(i)*u);
			fu = errorfunction.error(model, in, target);
		}
		ax = bx;
		bx = cx;
		cx = u;
		fa = fb;
		fb = fc;
		fc = fu;
	}
	//=======================================================================

	e = 0.;
	if (ax < cx)
	{
		a = ax;
		b = cx;
	}
	else
	{
		a = cx;
		b = ax;
	}
	x = _w = v = bx;
	for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*x);
	fw = fv = fx = errorfunction.error(model, in, target);

	for (iter = 0; iter < ITMAX; iter++)
	{
		xm = 0.5 * (a + b);
		tol2 = 2. * (tol1 = TOL * fabs(x) + ZEPS);
		if (fabs(x - xm) <= (tol2 - 0.5 *(b - a)))
		{
			break;
		}
		if (fabs(e) > tol1)
		{
			r = (x - _w) * (fx - fv);
			q = (x - v) * (fx - fw);
			o = (x - v) * q - (x - _w) * r;
			q = 2. * (q - r);
			if (q > 0.)
				o = -o;
			q = fabs(q);
			etemp = e;
			e = d;
			if (fabs(o) >= fabs(0.5 * q * etemp) ||
					o <= q *(a - x) ||
					o >= q *(b - x))
				d = CGOLD * (e = (x >= xm ? a - x : b - x));
			else
			{
				d = o / q;
				u = x + d;
				if (u - a < tol2 || b - u < tol2)
					d = SIGN(tol1, xm - x);
			}
		}
		else
			d = CGOLD * (e = (x >= xm ? a - x : b - x));
		u = (fabs(d) >= tol1 ? x + d : x + SIGN(tol1, d));
		for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*u);
		fu = errorfunction.error(model, in, target);
		if (fu <= fx)
		{
			if (u >= x)
				a = x;
			else
				b = x;
			v = _w;
			_w = x;
			x = u;
			fv = fw;
			fw = fx;
			fx = fu;
		}
		else
		{
			if (u < x)
				a = u;
			else
				b = u;
			if (fu <= fw || _w == x)
			{
				v = _w;
				_w = u;
				fv = fw;
				fw = fu;
			}
			else if (fu <= fv || v == x || v == _w)
			{
				v = u;
				fv = fu;
			}
		}
	}

	if (iter >= ITMAX)
		throw SHARKEXCEPTION("BFGS: too many iterations in brent");
	fret = fx;
	//=======================================================================

	if (fx < initial)
	{
		fret = fx;
		for (i = 0; i < n; ++i)
			p(i) += xi(i) * x;
	}
	else
	{
		fret = initial;
	}
}
//======================================================================
// Definition of the cubic line search class
// Armijo and Goldstein's line search algorithm

// author:  Doug Hart, Adapted to the COOOL library by Wenceslau Gouveia

// Modified to fit into new classes.  H. Lydia Deng, 02/21/94, 03/15/94

//========================================================================


/*

 * cubic line search

 *

 * The parameter lambda controls the accuraccy of the line search.

 * lambda = .25 is a good choice.

 */

void BFGS::cblnsrch(
	Model& model,
	ErrorFunction& errorfunction,
	double fold,
	Array< double >& g,
	Array< double >& p,
	double& f,
	const Array< double >&in,
	const Array< double >&target)
{
	//
	// the maximum of performed iterations should be choosen small enough
	// to ensure that neither (3^iterMax) nor (0.1^(2*iterMax)) causes a
	// numerical overflow or underflow.
	//
	const unsigned iterMax = 20;
	unsigned iterNum;
	unsigned i, n;
	bool     tst = false;
	double   slope;
	double   alpha, alpha2;
	double   alpha_prev, alpha_prev2;
	double   alpha_tmp = 0;
	double   f1, f2, fprev = 0;
	double   a, b, c;
	double   cm11, cm12, cm21, cm22;
	double   disc;
	double   lambda = LS_lambda;

	n = BFGS_n;
	/*

	 * dot product of search direction and gradient

	 */

	for (slope = 0., i = 0; i < n; i++)
		slope += g(i) * p(i);

	iterNum = 0;			/* iteration counter */
	alpha   = 1.;			/* updating step */

	/*
	 * updating
	 */

	for (i = 0; i < n; i++)
		model.setParameter(i, model.getParameter(i) + alpha * p(i));
	f = errorfunction.error(model, in, target);
	iterNum++;
	/*
	 * Implementing Goldstein's test for alpha too small
	 */
	while (f < fold + (1. - lambda)*alpha*slope && iterNum < iterMax)
	{
		alpha *= 3;
		for (i = 0; i < n; i++)
			model.setParameter(i, model.getParameter(i) + alpha * p(i));
		f = errorfunction.error(model, in, target);
		iterNum++;
	}
	if (iterNum >= iterMax)
		throw SHARKEXCEPTION("BFGS: alpha overflowed!");

	/*
	 * Armijo's test for alpha too large
	 */

	alpha_prev = alpha; /* H.L. Deng, 6/13/95 */
	while (f > fold + lambda*alpha*slope && iterNum < iterMax)
	{
		alpha2 = alpha * alpha;
		f1 = f - fold - slope * alpha;
		if (tst == false)
		{
			alpha_tmp = -slope * alpha2 / (f1 * 2.); /* tentative alpha */
			tst = true;
		}
		else
		{
			alpha_prev2 = alpha_prev * alpha_prev;
			f2   = fprev - fold - alpha_prev * slope;
			c    = 1. / (alpha - alpha_prev);
			cm11 = 1. / alpha2;
			cm12 = -1. / alpha_prev2;
			cm21 = -alpha_prev / alpha2;
			cm22 = alpha / alpha_prev2;
			a    = c * (cm11 * f1 + cm12 * f2);
			b    = c * (cm21 * f1 + cm22 * f2);
			disc = b * b - 3. * a * slope;

			if ((fabs(a) > FLT_MIN) && (disc > FLT_MIN))
				alpha_tmp = (-b + sqrt(disc)) / (3. * a);
			else
				alpha_tmp = slope * alpha2 / (2. * f1);
			if (alpha_tmp >= .5 * alpha)
				alpha_tmp = .5 * alpha;
		}
		alpha_prev = alpha;
		fprev = f;
		if (alpha_tmp < .1 * alpha)
			alpha *= .1;
		else
			alpha = alpha_tmp;
		for (i = 0; i < n; i++)
			model.setParameter(i, model.getParameter(i) + alpha * p(i));
		f = errorfunction.error(model, in, target);
		iterNum++;
	}
	if (iterNum >= iterMax)
		throw SHARKEXCEPTION("BFGS: alpha underflowed!");
}

