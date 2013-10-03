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

#ifndef SHARK_ALGORITHMS_GRADIENTDESCENT_IMPL_WOLFECUBIC_INL
#define SHARK_ALGORITHMS_GRADIENTDESCENT_IMPL_WOLFECUBIC_INL

namespace shark {

namespace detail{
///  \briefCubic interpolation from Numerical Optimization p. 59.
inline double wlsCubicInterp(double t1, double t2,
        double f1, double f2,
        double gtd1, double gtd2
) {
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

	// New step length should be in interval [t1, t2]
	return std::min(std::max(t, t1), t2);
}
/// \brief Line search, using cubic interpolation, satisfying the strong Wolfe conditions.
template <class VectorT, class Function>
void wolfecubic(
	VectorT &point,
	const VectorT &searchDirection,
	double &value,
	Function const& func,
	VectorT &gradient,
	double t = 1.0
) {
	SIZE_CHECK(point.size() == searchDirection.size());
	SIZE_CHECK(point.size() == gradient.size());

	// Constants
	const double tol = 1e-9;
	const size_t maxIter = 25;
	const double c1 = 1e-4;
	const double c2 = 0.9;
	double maxD = 0.0;
	for (size_t i = 0; i < searchDirection.size(); ++i)
		maxD = std::max(maxD, std::abs(searchDirection(i)));

	// Previous step
	double f_prev = value;
	double t_prev = 0.0;
	VectorT g_prev = gradient;
	double gtd = inner_prod(gradient, searchDirection);

	// Initial step values
	VectorT g_new(point.size());
	double f_new  = func.evalDerivative(point + t * searchDirection, g_new);
	double gtd_new = inner_prod(g_new, searchDirection);

	// Bracket vars
	double bracket[2];//bracket [t_lo,t_hi]
	double bracketf[2];//function values at the borders of the bracket
	VectorT bracketg[2];//gradient values at the borders of the bracket
	bool single = false;

	unsigned int iter = 0;
	bool done = false;

	// Bracketing phase.
	// Create a bracke [t_lo, t_hi] within which there lies a point satisfying
	// the strong Wolfe conditions with certainty.
	while(iter++ < maxIter) {
		// If the new point doesn't decrease enough there must be one that
		// does so before it.
		if (f_new > value + c1 * t * gtd || (iter > 1 && f_new >= f_prev)) {
			bracket[0] = t_prev;
			bracket[1] = t;
			bracketf[0] = f_prev;
			bracketf[1] = f_new;
			bracketg[0] = g_prev;
			bracketg[1] = g_new;
			break;
		}
		// If t satisfies both wolfe conditions we are done.
		if (std::abs(gtd_new) <= -c2 * gtd) {
			bracket[0] = t;
			bracketf[0] = f_new;
			bracketg[0] = g_new;
			done = single = true;
			break;
		}
		// If the directional searchDirection at the new point is positive, there
		// must be a good point before it.
		if (gtd_new >= 0) {
			bracket[0] = t_prev;
			bracket[1] = t;
			bracketf[0] = f_prev;
			bracketf[1] = f_new;
			bracketg[0] = g_prev;
			bracketg[1] = g_new;
			break;
		}

		t_prev = t;
		t *= 10; // Expand the bracket.

		f_prev = f_new;
		g_prev = g_new;
		f_new = func.evalDerivative(point + t * searchDirection, g_new);
		gtd_new = inner_prod(g_new, searchDirection);
	}

	// Convention: prev = lo and new = hi.
	bool insuf = false;
	// Zooming phase
	// We know a good point lies in [t_lo, t_hi]. Now find it!
	while (!done && iter++ < maxIter) {
		// Make sure lo and hi are correct
		size_t lo = bracketf[1] < bracketf[0] ? 1 : 0;
		size_t hi = 1 - lo;

		// Cubic interpolation for new step length.
		t = wlsCubicInterp(bracket[0], bracket[1], bracketf[0], bracketf[1],
		        inner_prod(bracketg[0], searchDirection),
		        inner_prod(bracketg[1], searchDirection));

		// Sufficient progress?
		double mint = std::min(bracket[0], bracket[1]);
		double maxt = std::max(bracket[0], bracket[1]);
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

		f_new = func.evalDerivative(point + t * searchDirection, g_new);
		gtd_new = inner_prod(g_new, searchDirection);

		// If the new point doesn't decrease enough, make it new t_hi
		if (f_new > value + c1 * t * gtd || f_new > bracketf[lo]) {
			bracket[hi] = t;
			bracketf[hi] = f_new;
			bracketg[hi] = g_new;
		} else {
			// If step has sufficient length, use it.
			if (std::abs(gtd_new) <= -c2 * gtd)
				done = true;
			else if (gtd_new * (bracket[hi] - bracket[lo]) >= 0) {
				bracket[hi] = bracket[lo];
				bracketf[hi] = bracketf[lo];
				bracketg[hi] = bracketg[lo];
			}
			bracket[lo] = t;
			bracketf[lo] = f_new;
			bracketg[lo] = g_new;
		}

		// tolerance checking. If the steps become too small avoid them.
		if (!done && std::abs(bracket[0] - bracket[1])*maxD < tol) {
			break;
		}
	}

	if (iter < maxIter || value > bracketf[0] || value > bracketf[1]) {
		if (bracketf[0] < bracketf[1] || single) {
			point += bracket[0] * searchDirection;
			value = bracketf[0];
			gradient = bracketg[0];
		} else {
			point += bracket[1] * searchDirection;
			value = bracketf[1];
			gradient = bracketg[1];
		}
	}
}

}}


#endif
