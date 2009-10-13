/*!
 *  \file CG.cpp
 *
 *  \brief Offers the Conjugate Gradients algorithm
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
 */


#include <cmath>
#include <iostream>
#include <sstream>
#include <float.h>
#include <SharkDefs.h>
#include <Array/ArrayOp.h>
#include <ReClaM/CG.h>


using namespace std;


//===========================================================================
/*!
 *  \brief Initializes some internal variables used by the CG algorithm.
 *
 *  The given parameters are used to initialize some variables
 *  used by the CG and the line search algorithms and to allocate
 *  memory for internal structures.
 *
*	\param model	  The currently used model including the parameter vector.
 *      \param ls         Type of LineSearch algorithm, that is used
 *                        by the CG algorithm. Possible values:
 *                        "Dlinmin" (default value), "Linmin", "Cblnsrch"
 *      \param reset      No. of iterations that will be performed before
 *                        a reset of the algorithm takes place. A reset
 *                        is sometimes necessary because often the
 *                        line minimization has not finished after
 *                        \f$n = \f$ "no. of parameters" iterations.
 *                        If set to "0" (default
 *                        value), then the algorithm will be reset after
 *                        \f$n = |w| + 1\f$ iterations, where \f$w\f$
 *                        is the parameter vector.
 *      \param ax         Initial left bracket for the line search
 *                        algorithms, the default bracket is "0".
 *      \param bx         Initial right bracket for the line search
 *                        algorithms, the default bracket is "1".
 *      \param lambda     The initial value for the searched \f$\lambda\f$,
 *                        that will minimize the function used by the
 *                        line search algorithm. The default value is "0.25".
 *      \param verbose    If set to "true", a message will displayed
 *                        when the CG algorithm performs a reset.
 *      \return none
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes
 *	2006-08-14,   : <br>
 *	Addad parameter "model" and changed name from "initCG" to "init",
 * 	because of new shark architecture.
 *      2002-01-16, ra: <br>
 *      Removed parameters "eps" and "ftol", because they are
 *      not used.
 *
 */
void CG::init(Model& model,
			  LineSearch ls,
			  int reset,
			  double ax,
			  double bx,
			  double lambda,
			  bool verbose)
{
	first_iteration = true;
	lineSearchType = ls;
	LS_lambda      = lambda;
	LS_ax          = ax;
	LS_bx          = bx;
	CG_verbose     = verbose;
	CG_n = model.getParameterDimension();

	if (reset >= 0)
		CG_reset       = reset;
	else
		CG_reset       = CG_n + 1;

	CG_count       = 0;
	CG_g .resize(CG_n, false);
	CG_h .resize(CG_n, false);
	CG_xi.resize(CG_n, false);
}

void CG::init(Model& model)
{
	init(model, Dlinmin);
}

//===========================================================================
/*!
 *  \brief Updates the parameters of the model by using
 *         the CG optimization algorithm.
 *
 *  The given input patterns \em in and their corresponding target
 *  values \em target are used to calculate the error using the given \em errorfunction of the currently
 *  given \em model . The resulting error function is then minimized by the
 *  CG algorithm, the parameters of the model are updated. <br>
 *  Notice: Method #init must be called before using the CG
 *  algorithm itself.
 *
 *	\param model		the currently used model including the parameter vector
 *	\param errorfunction	function used to calculate the error of current model
 *      \param in		Input patterns for the currently used model.
 *      \param target		Target values corresponding to the input patterns.
 *      \return			the minimum value of the error
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes
 *	2006-08-14,   : <br>
 *	Addad parameter "model" and "errorfunction",
 * 	changed name from "cg" to "optimize", because of new shark architecture.
 *
 *  \par Status
 *      stable
 *
 */

double CG::optimize(
	Model& model,
	ErrorFunction& errorfunction,
	const Array<double>& input,
	const Array<double>& target)
{
	double fret;
	Array<double> dedw(CG_n);

//	part of init method
	if (first_iteration)
	{
		CG_fret = errorfunction.errorDerivative(model, input, target, dedw);
		for (unsigned j = 0; j < CG_n; j++)
		{
			CG_xi(j) = CG_h(j) = CG_g(j) = -dedw(j);
		}
		first_iteration = false;
	}
	unsigned j;

	Array<double> CG_p(CG_n);
	for (j = 0; j < CG_n; j++) CG_p(j) = model.getParameter(j);

	fret = CG_fret;

#ifdef PARANOID
	double before = errorfunction.error(model, input, target);
#endif

	switch (lineSearchType)
	{
	case Dlinmin:
		dlinmin(model, errorfunction, dedw, CG_p, CG_xi, fret, input, target);
		break;
	case Linmin:
		linmin(model, errorfunction, dedw, CG_p, CG_xi, fret, input, target);
		break;
	case Cblnsrch:
		cblnsrch(model, errorfunction, CG_fret, dedw, CG_xi, CG_fret, input, target);
		for (j = 0; j < CG_n; j++) CG_p(j) = model.getParameter(j);
		break;
	default:
		stringstream s;
		s << "line search type " << lineSearchType << " not implemented in conjugate gradient method" << std::endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}

	for (j = 0; j < CG_n; j++) model.setParameter(j, CG_p(j));
	CG_fret = fret = errorfunction.errorDerivative(model, input, target, dedw);

#ifdef PARANOID

	if (fret > before)
	{
		stringstream s;
		s << "error increase from " << fret << " to " << before << " observed in paranoid check - this should never ever happen" << std::endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}

#endif

	CG_xi = dedw;
	CG_dgg = CG_gg = 0.0;
	for (j = 0; j < CG_n; j++)
	{
		CG_gg += CG_g(j) * CG_g(j);
		CG_dgg += (CG_xi(j) + CG_g(j)) * CG_xi(j);
	}

	if (CG_gg == 0.0) return fret;		// new 10/2008: converged

	CG_gam = CG_dgg / CG_gg;
	for (j = 0; j < CG_n;j++)
	{
		CG_g(j) = -CG_xi(j);
		CG_xi(j) = CG_h(j) = CG_g(j) + CG_gam * CG_h(j);
	}

	CG_gg = 0.;
	for (j = 0; j < CG_n;j++)
	{
		CG_gg += CG_xi(j) * -dedw(j);
	}
	if (CG_gg <= 0.)
	{
		CG_xi = CG_h = CG_g = -dedw;
		if (CG_verbose) std::cerr << "CG: reset direction to gradient because gradient and conjugate gradient differ by more than 90 degrees" << std::endl;
	}

	CG_count++;
	if (CG_count == CG_reset)
	{
		reset(dedw);
		//CG_count = 0;
		//CG_xi= CG_h = CG_g = -dedw;
		if (CG_verbose) std::cerr << "CG: regular reset of search direction to gradient" << std::endl;
	}
	//Returns fret
	return fret;
}

void CG::reset(Array< double >& dedw)
{
	CG_count = 0;
	dedw.resize(CG_n, true);
	for (unsigned i = 0;i < CG_n;i++)CG_xi = CG_h(i) = CG_g(i) = -dedw(i);
}



#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))


void CG::dlinmin(
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

	unsigned        n = CG_n;
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
	//TODO exeptions
	if (iter >= ITMAX)
		std::cerr << "Too many iterations in routine dbrent" << std::endl;
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

void CG::linmin(Model& model,
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

	unsigned i, iter, n = CG_n;

	double   fa, fb, fc, ax, bx, cx;
	double   ulim, dum;
	double   a, b, d = 0., e, etemp, fu, fv, fw, fx, o, q, r, tol1, tol2, u, v, _w, x, xm;

	Array< double > xt(n);

	ax = LS_ax;
	bx = LS_bx;

	//===================================================================

	for (i = 0; i < n; ++i) model.setParameter(i, p(i) + xi(i)*ax);
	double initial = fa = errorfunction.error(model, in, target);

	//cout << "initial: " << initial << endl;

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

	//TODO exeptions
	//ASSERT( iter < ITMAX )

	if (iter >= ITMAX)
		std::cerr << "Too many iterations in brent" << std::endl;

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

void CG::cblnsrch(
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
	unsigned i, n = CG_n;

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

	for (i = 0; i < n; i++)model.setParameter(i, model.getParameter(i) + alpha * p(i));
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
	// TODO exeption
	if (iterNum >= iterMax)
		std::cerr << "Alpha over flowed!" << std::endl;

	//cerr << "alpha = " << alpha << "\titerNum = " << iterNum << endl;

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
		for (i = 0; i < n; i++)model.setParameter(i, model.getParameter(i) + alpha * p(i));
		f = errorfunction.error(model, in, target);
		iterNum++;
	}

	//TODO exeptions

	if (iterNum >= iterMax)
		std::cerr << "Alpha under flowed!" << std::endl;
}

