//===========================================================================
/*!
 *  \file cblnsrch.cpp
 *
 *  \brief Cubic line search algorithm for finding a
 *         minimum value of a function.
 *
 *  \author  M. Kreutz
 *  \date    1995
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
 */
 
 /*
 *  The algorithm was taken from the library:
 *
 *  ============================================================<BR>
 *  COOOL           version 1.1           ---     Nov,  1995    <BR>
 *  Center for Wave Phenomena, Colorado School of Mines         <BR>
 *  ============================================================<BR>
 *
 *  This code is part of a preliminary release of COOOL (CWP
 *  Object-Oriented Optimization Library) and associated class
 *  libraries.
 *
 *  The COOOL library is a free software. You can do anything you want
 *  with it including make a fortune.  However, neither the authors,
 *  the Center for Wave Phenomena, nor anyone else you can think of
 *  makes any guarantees about anything in this package or any aspect
 *  of its functionality.
 *
 *  Since you've got the source code you can also modify the
 *  library to suit your own purposes. We would appreciate it
 *  if the headers that identify the authors are kept in the
 *  source code.
 *
 *  ======================================================================<BR>
 *  Definition of the cubic line search class                             <BR>
 *  Armijo and Goldstein's line search algorithm                          <BR>
 *  author:  Doug Hart, Adapted to the COOOL library by Wenceslau Gouveia <BR>
 *  Modified to fit into new classes.  H. Lydia Deng, 02/21/94, 03/15/94  <BR>
 *  ======================================================================<BR>
 *
 */
//===========================================================================


#include <cmath>
#include <climits>
#include <cfloat>
#include <SharkDefs.h>
#include <LinAlg/arrayoptimize.h>


//===========================================================================
/*!
 *  \brief Does a cubic line search, i.e. given a nonlinear function,
 *         a starting point and a direction,
 *         a new point is calculated where the function has
 *         decreased "sufficiently".
 *
 *  The line search algorithms are based on the Newton method of
 *  approximating root values of monotone decreasing
 *  functions. When the derivative \f$ f' \f$ of the function \f$ f \f$
 *  at a starting
 *  point \f$ x \f$ on the X-axis can be calculated, the intersection
 *  \f$ x' \f$ of the tangent at point \f$ f(x) \f$ (with gradient
 *  \f$ f'(x) \f$) with the X-axis can be used to get a better approximation
 *  of the minima at \f$ x_{min} \f$.
 *
 *  This function is based on this idea: Given a nonlinear function
 *  \f$ f \f$, a n-dimensional starting point
 *  \f$ x_{old} \f$ and a direction \f$ p \f$ (known as
 *  Newton direction), a new point \f$ x_{new} \f$ is calculated
 *  as
 *
 *  \f$
 *      x_{new} = x_{old} + \lambda p, \hspace{2em} 0 < \lambda \leq 1
 *  \f$
 *
 *  in a way that \f$ f(x_{new})\f$ has decreased sufficiently.
 *  Sufficiently means that
 *
 *  \f$
 *      f(x_{new}) \leq f(x_{old}) + \alpha \nabla f \cdot (x_{new} - x_{old})
 *  \f$
 *
 *  where \f$ 0 < \alpha < 1 \f$ is a fraction of the initial
 *  rate of decrease \f$ \nabla f \cdot p \f$.
 *
 *  This function can be used for minimization or solving a set of
 *  nonlinear equations of the form \f$ F(x) = 0 \f$.
 *  Finding the root value (the x-value at which the related function
 *  will intersect the X-axis) will then solve the equations.
 *  In contrast to line search (lnsrch.cpp) the cubic line search interpolates
 *  four points to find the minimum value and is useful, when
 *  gradient information is already available or when more than
 *  three function evaluations have been calculated.
 *
 *  \param xold   n-dimensional starting point.
 *  \param fold   The value of function \em func at point \em xold.
 *  \param g      The n-dimensional gradient of function \em func at point
 *                \em xold.
 *  \param p      n-dimensional point to specify the search direction (the
 *                Newton direction).
 *  \param x      New n-dimensional point along the direction \em p
 *                from \em xold where the function \em func decreases
 *                "sufficiently".
 *  \param f      The new function value for point \em x.
 *  \param func   Function to decrease.
 *  \param lambda Controls the accuracy of line search, \f$ 0.25 \f$ is a
 *                good choice.
 *  \return       none.
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
 *  \sa lnsrch.cpp
 *
 */
void cblnsrch
(
	Array< double >& xold,
	double fold,        /* evaluation at the current solution */
	Array< double >& g, /* current gradient */
	Array< double >& p, /* search direction */
	Array< double >& x, /* returns the new solution */
	double& f,
	double(*func)(const Array< double >&),
	double lambda
)
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

	n = xold.nelem();

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
		x(i) = xold(i) + alpha * p(i);
	f = func(x);
	iterNum++;

	/*
	 * Implementing Goldstein's test for alpha too small
	 */
	while (f < fold + (1. - lambda)*alpha*slope && iterNum < iterMax) {
		alpha *= 3;
		for (i = 0; i < n; i++)
			x(i) = xold(i) + alpha * p(i);
		f = func(x);
		iterNum++;
	}
	if (iterNum >= iterMax)
		std::cerr << "Alpha over flowed!" << std::endl;

	//std::cerr << "alpha = " << alpha << "\titerNum = " << iterNum << std::endl;

	/*
	 * Armijo's test for alpha too large
	 */
	alpha_prev = alpha; /* H.L. Deng, 6/13/95 */
	while (f > fold + lambda*alpha*slope && iterNum < iterMax) {
		alpha2 = alpha * alpha;
		f1 = f - fold - slope * alpha;

		if (tst == false) {
			alpha_tmp = -slope * alpha2 / (f1 * 2.); /* tentative alpha */
			tst = true;
		}
		else {
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
			x(i) = xold(i) + alpha * p(i);
		f = func(x);
		iterNum++;
	}
	if (iterNum >= iterMax)
		std::cerr << "Alpha under flowed!" << std::endl;

	//std::cerr << "alpha = " << alpha << "\titerNum = " << iterNum << std::endl;
}

