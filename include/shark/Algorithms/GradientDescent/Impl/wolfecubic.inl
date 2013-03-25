//===========================================================================
/*!
 *  \brief Line search satisfying the Strong Wolfe conditions
 *
 *  Classic bracket-zoom line search procedure, which produces a step length
 *  satisfying the strong Wolfe conditions. This method is described in
 *  Numerical Optimization by Nocedal and Wright in chapter 3.5.
 *  The procedure works by first finding an interval (a1, a2), which contains
 *  an acceptable step length. Secondly this interval is decreased until we
 *  find an acceptable step length.
 *  The method uses cubic interpolation in the zooming phase.
 *
 *  \author S. Dahlgaard
 *  \date   2012
 *
 *
 *  \par Copyright (c) 1998-2000:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      searchDirection-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
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

#ifndef SHARK_LINESEARCH_WOLFECUBIC_INL
#define SHARK_LINESEARCH_WOLFECUBIC_INL

#include <iostream>

namespace shark {

double wlsCubicInterp(double t1, double t2,
        double f1, double f2,
        double gtd1, double gtd2);
//! \brief Line search, using cubic interpolation, satisfying the strong Wolfe conditions
template <class VectorT, class Function>
void wolfecubic
(
    VectorT &point,
    const VectorT &searchDirection,
    double &fret,
    Function func,
    VectorT const &gradient
) {
	SIZE_CHECK(point.size() == searchDirection.size());
	SIZE_CHECK(point.size() == gradient.size());

	// Constants
	const double tol = 1e-9;
	const size_t maxIter = 25;
	const double c1 = 1e-4;
	const double c2 = 0.9;

	// Previous step
	double gtd = inner_prod(gradient, searchDirection);
	double f_prev = fret;
	double t_prev = 0.0;
	double gtd_prev = gtd;

	// Initial step values
	double t = 1.0;
	VectorT g_new(point.size());
	double f_new  = func(point + t * searchDirection, g_new);
	double gtd_new = inner_prod(g_new, searchDirection);

	// Bracket vars
	double t_lo, t_hi;
	double f_lo, f_hi;
	double gtd_lo, gtd_hi;

	unsigned int iter = 0;
	bool done = false;

	// The answer.
	double t_ans = 0.0;
	double f_ans = fret;

	// Bracketing phase.
	// Create a bracke [t_lo, t_hi] within which there lies a point satisfying
	// the strong Wolfe conditions with certainty.
	while (iter++ < maxIter) {
		// If the new point doesn't decrease enough there must be one that
		// does so before it.
		if (f_new > fret + c1 * t * gtd || (iter > 1 && f_new >= f_prev)) {
			t_lo = t_prev;
			f_lo = f_prev;
			gtd_lo = gtd_prev;
			t_hi = t;
			f_hi = f_new;
			gtd_hi = gtd_new;
			break;
		}
		// If t satisfies both wolfe conditions we are done.
		if (std::abs(gtd_new) <= -c2 * gtd) {
			t_ans = t;
			f_ans = f_new;
			done = true;
			break;
		}
		// If the directional searchDirection at the new point is positive, there
		// must be a good point before it.
		if (gtd_new >= 0) {
			t_hi = t_prev;
			f_hi = f_prev;
			gtd_hi = gtd_prev;
			t_lo = t;
			f_lo = f_new;
			gtd_lo = gtd_new;
			break;
		}

		t_prev = t;
		t *= 10; // Expand the bracket.

		f_prev = f_new;
		gtd_prev = gtd_new;
		f_new = func(point + t * searchDirection, g_new);
		gtd_new = inner_prod(g_new, searchDirection);
	}

	// Convention: prev = lo and new = hi.
	bool insuf = false;
	// Zooming phase
	// We know a good point lies in [t_lo, t_hi]. Now find it!
	while (!done && iter++ < maxIter) {
		// Make sure lo and hi are correct
		double mint = std::min(t_lo, t_hi);
		double maxt = std::max(t_lo, t_hi);

		// Cubic interpolation for new step length.
		t = wlsCubicInterp(t_lo, t_hi, f_lo, f_hi, gtd_lo, gtd_hi);

		// Sufficient progress?
		if (std::min(maxt - t, t - mint)/(maxt - mint) < 0.1) {
			if (insuf || t >= maxt || t <= mint) {
				if (std::abs(t - maxt) < std::abs(t - mint))
					t = maxt - 0.1 * (maxt - mint);
				else
					t = mint + 0.1 * (maxt - mint);
				insuf = false;
			} else
				insuf = true;
		} else
			insuf = false;

		f_new = func(point + t * searchDirection, g_new);
		gtd_new = inner_prod(g_new, searchDirection);

		// If the new point doesn't decrease enough, make it new t_hi
		if (f_new > fret + c1 * t * gtd || f_new > f_lo) {
			t_hi = t;
			f_hi = f_new;
			gtd_hi = gtd_new;
		} else {
			// If step has sufficient length, use it.
			if (std::abs(gtd_new) <= -c2 * gtd) {
				t_ans = t;
				f_ans = f_new;
				done = true;
			}
			// Else make point the new t_lo. This may violate the condition
			// that gtd_lo * (t_hi - t_lo) < 0, if so swap hi with lo first.
			else if (gtd_new * (t_hi - t_lo) >= 0) {
				t_hi = t_lo;
				f_hi = f_lo;
				gtd_hi = gtd_lo;
			}
			t_lo = t;
			f_lo = f_new;
			gtd_lo = gtd_new;
		}

		// tolerance checking. If the steps become too small avoid them.
		if (std::abs((t_lo - t_hi)*gtd_new) < tol) {
			break;
		}
	}

	// Adjust the return values.
	point += t_ans * searchDirection;
	fret = f_ans;
}

// Cubic interpolation from Numerical Optimization p. 59.
double wlsCubicInterp(double t1, double t2,
        double f1, double f2,
        double gtd1, double gtd2) {
	if (t1 == t2)
		return t1;
	if (t2 < t1) {
		std::swap(t1, t2);
		std::swap(f1, f2);
		std::swap(gtd1, gtd2);
	}

	double d1 = gtd1 + gtd2 - 3 * (f1 - f2)/(t1 - t2);
	if (d1 * d1 - gtd1 * gtd2 < 0)
		return (t1 + t2) / 2.0;
	double d2 = std::sqrt(d1 * d1 - gtd1 * gtd2);
	double t = t2 - (t2 - t1) * ((gtd2 + d2 - d1)/(gtd2 - gtd1 + 2 * d2));

	// New step length should be in intervel [t1, t2]
	return std::min(std::max(t, t1), t2);
}

}


#endif // SHARK_LINESEARCH_WOLFECUBIC_INL